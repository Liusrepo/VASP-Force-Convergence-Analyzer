#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VASP Force Convergence Analyzer
Author: Leo; 
"""

import re 
import math 
import sys
import os
import datetime
import json
import time
from collections import Counter


class VASPForceAnalyzer:
    
    def __init__(self, poscar_file="POSCAR", incar_file="INCAR", outcar_file="OUTCAR"):
        self.poscar_file = poscar_file
        self.incar_file = incar_file
        self.outcar_file = outcar_file
        self.atom_data = []
        self.force_tables = []
        self.ediffg = 0.03
        self.log_data = {"timestamp": datetime.datetime.now().isoformat()}

    def parse_incar(self):
        try:
            with open(self.incar_file, 'r') as f:
                for line in f:
                    if 'EDIFFG' in line and not line.strip().startswith('#'):
                        match = re.search(r'(?i)\bEDIFFG\s*=\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eEdD][+-]?\d+)?)', line)
                        if match:
                            val = match.group(1).replace('D','E').replace('d','E')
                            self.ediffg = abs(float(val))
                            self.log_data["ediffg"] = self.ediffg
                            return self.ediffg
        except IOError:
            print("Warning: INCAR file not found, using default EDIFFG = {0} eV/A".format(self.ediffg))
        return self.ediffg

    def parse_poscar(self):
        try:
            with open(self.poscar_file, 'r') as f:
                lines = [ln.rstrip('\n') for ln in f]

            if len(lines) < 8:
                raise ValueError("POSCAR seems too short.")

            # Parse elements and counts (VASP5 format preferred)
            elements = lines[5].split()
            counts_line = lines[6].split()
            try:
                counts = [int(x) for x in counts_line]
            except ValueError:
                # VASP4 compatibility
                elements = ['X']
                try:
                    counts = [int(x) for x in lines[5].split()]
                except Exception:
                    raise ValueError("Failed to parse element counts from POSCAR.")
                counts = [sum(counts)]

            if len(elements) != len(counts):
                raise ValueError(f"Elements ({elements}) and counts ({counts}) length mismatch.")

            total_atoms = sum(counts)

            # Locate selective dynamics and coordinate system lines
            selective_idx = -1
            for i, line in enumerate(lines):
                if line.strip().lower().startswith('selective'):
                    selective_idx = i
                    break

            coord_type_idx = selective_idx + 1 if selective_idx != -1 else 7
            if coord_type_idx >= len(lines):
                raise ValueError("Failed to locate coordinate system line (Direct/Cartesian).")

            coord_type_str = lines[coord_type_idx].strip().lower()
            if coord_type_str.startswith('d'):
                coord_type = 'direct'
            elif coord_type_str.startswith('c'):
                coord_type = 'cartesian'
            else:
                if coord_type_idx + 1 < len(lines):
                    s2 = lines[coord_type_idx + 1].strip().lower()
                    if s2.startswith('d'):
                        coord_type = 'direct'
                        coord_type_idx += 1
                    elif s2.startswith('c'):
                        coord_type = 'cartesian'
                        coord_type_idx += 1
                    else:
                        raise ValueError("Unknown coordinate system; expected 'Direct' or 'Cartesian'.")
                else:
                    raise ValueError("Unknown coordinate system; expected 'Direct' or 'Cartesian'.")

            coord_start = coord_type_idx + 1
            self.log_data["coord_type"] = coord_type

            # Build atom list
            atom_data = []
            atom_idx = 0
            for elem, count in zip(elements, counts):
                for _ in range(count):
                    atom_idx += 1
                    atom_data.append({
                        'index': atom_idx,
                        'element': elem,
                        'label': f"{elem}{atom_idx}",
                        'position': None,
                        'movable': True,
                        'constraints': (True, True, True)
                    })

            # Parse coordinates and selective dynamics flags
            for i, atom in enumerate(atom_data):
                line_idx = coord_start + i
                if line_idx >= len(lines):
                    break
                line = lines[line_idx].strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) < 3:
                    continue
                try:
                    x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                except ValueError:
                    continue

                atom['position'] = (x, y, z)

                if len(parts) >= 6:
                    tx = parts[3].upper().startswith('T')
                    ty = parts[4].upper().startswith('T')
                    tz = parts[5].upper().startswith('T')
                    atom['movable'] = tx or ty or tz
                    atom['constraints'] = (tx, ty, tz)
                else:
                    if selective_idx != -1:
                        atom['movable'] = True
                        atom['constraints'] = (True, True, True)

            self.atom_data = atom_data
            self.log_data["total_atoms"] = len(atom_data)
            self.log_data["movable_atoms"] = sum(1 for a in atom_data if a['movable'])

            # Cache movable atom indices for performance
            self._movable_idx = [i for i, a in enumerate(atom_data) if a['movable']]

            if any(a['position'] is None for a in atom_data):
                missing = sum(1 for a in atom_data if a['position'] is None)
                print(f"Warning: {missing} atom(s) have no parsed position in POSCAR (line count may be short).")

            return atom_data

        except IOError:
            print("Warning: POSCAR file not found")
            return []
        except ValueError as e:
            print(f"Error: {e}")
            return []

    def parse_outcar(self):
        """Parse force data from OUTCAR file."""
        HEADER_RE = re.compile(r'^\s*POSITION\b.*TOTAL-FORCE', re.I)
        DASH_RE = re.compile(r'^-+')
        
        try:
            force_tables = []
            forces = None
            with open(self.outcar_file, 'r') as f:
                in_block = False
                skip_next = 0

                for raw in f:
                    line = raw.rstrip('\n')
                    if not in_block:
                        if HEADER_RE.search(line):
                            in_block = True
                            skip_next = 1
                            forces = []
                        continue

                    if skip_next > 0:
                        skip_next -= 1
                        continue

                    s = line.strip()
                    if not s or DASH_RE.match(s):
                        if forces:
                            force_tables.append(forces)
                        in_block = False
                        forces = None
                        continue

                    parts = s.split()
                    if len(parts) >= 6:
                        try:
                            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                            fx, fy, fz = float(parts[3]), float(parts[4]), float(parts[5])
                            mag = math.hypot(fx, fy, fz)
                            forces.append({'position': (x,y,z), 'force': (fx,fy,fz), 'magnitude': mag})
                        except ValueError:
                            continue

            self.force_tables = force_tables
            self.log_data["ionic_steps"] = len(force_tables)
            return force_tables
        except IOError:
            print("Error: OUTCAR file not found!")
            return []

    def calculate_convergence_metrics(self):
        if not self.force_tables:
            return []
        
        convergence_data = []
        for step, forces in enumerate(self.force_tables, 1):
            if not forces:
                continue
            
            magnitudes = [f['magnitude'] for f in forces]
            max_force = max(magnitudes)
            rms_force = math.sqrt(sum(m**2 for m in magnitudes) / len(magnitudes))
            converged = max_force < self.ediffg
            
            convergence_data.append({
                'step': step,
                'max_force': max_force,
                'rms_force': rms_force,
                'converged': converged
            })
        
        return convergence_data

    def identify_problematic_atoms(self):
        """Identify atoms with forces exceeding EDIFFG threshold."""
        if not self.force_tables or not self.atom_data:
            return []
        
        final_forces = self.force_tables[-1]
        problematic_atoms = []
        
        movable_indices = getattr(self, "_movable_idx", range(len(self.atom_data)))
        
        for i in movable_indices:
            if i < len(final_forces) and i < len(self.atom_data):
                atom = self.atom_data[i]
                force_data = final_forces[i]
                if atom['movable'] and force_data['magnitude'] > self.ediffg:
                    problematic_atoms.append({
                        'index': atom['index'],
                        'element': atom['element'],
                        'position': atom['position'],
                        'force_magnitude': force_data['magnitude'],
                        'force_vector': force_data['force']
                    })
        
        problematic_atoms.sort(key=lambda x: x['force_magnitude'], reverse=True)
        return problematic_atoms

    def plot_convergence(self, width=80, height=20, char='*'):
        """Generate ASCII plot of force convergence on log scale."""
        data = self.calculate_convergence_metrics()
        if not data:
            print("Warning: No convergence data available for ASCII plotting")
            return

        series = [d['max_force'] for d in data]
        steps = [d['step'] for d in data]
        n = len(series)

        eps = 1e-12
        ediffg = max(self.ediffg, eps)
        vals = [max(v, eps) for v in series]

        # Log scale transformation
        yvals = [math.log10(v) for v in vals]
        yref = math.log10(ediffg)
        y_label = "log10(Force) [eV/Å]"

        ymin, ymax = min(yvals + [yref]), max(yvals + [yref])
        if abs(ymax - ymin) < 1e-12:
            ymax = ymin + 1.0

        left_margin = 10
        bottom_margin = 2
        plot_w = max(10, width - left_margin - 1)
        plot_h = max(6, height - bottom_margin - 1)

        canvas = [[' ' for _ in range(left_margin + plot_w + 1)]
                  for _ in range(plot_h + bottom_margin + 1)]

        # Draw axes
        y_axis_x = left_margin
        x_axis_y = plot_h
        for y in range(plot_h + 1):
            canvas[y][y_axis_x] = '|'
        for x in range(y_axis_x, y_axis_x + plot_w + 1):
            canvas[x_axis_y][x] = '-'
        canvas[x_axis_y][y_axis_x] = '+'

        def put_text(y, x, s):
            for i, ch in enumerate(s):
                if 0 <= y < len(canvas) and 0 <= x + i < len(canvas[0]):
                    canvas[y][x + i] = ch

        def fmt_val(v):
            val = 10 ** v
            return f"{val:.2e}"

        # Axis labels
        put_text(0, 0, f"{fmt_val(ymax):>9}")
        put_text(plot_h // 2, 0, f"{fmt_val((ymax + ymin) / 2):>9}")
        put_text(plot_h, 0, f"{fmt_val(ymin):>9}")

        put_text(x_axis_y + 1, y_axis_x, f"{steps[0]}")
        end_label = f"{steps[-1]}"
        put_text(x_axis_y + 1, y_axis_x + plot_w - len(end_label) + 1, end_label)

        def map_x(i):
            return y_axis_x + int(round(i * plot_w / (n - 1))) if n > 1 else y_axis_x + plot_w // 2

        def map_y(yval):
            t = (yval - ymin) / (ymax - ymin)
            t = max(0.0, min(1.0, t))
            return int(round((1 - t) * plot_h))

        # Draw threshold line
        y_thr = map_y(yref)
        for x in range(y_axis_x + 1, y_axis_x + plot_w + 1):
            if canvas[y_thr][x] == ' ':
                canvas[y_thr][x] = '='
        put_text(max(0, y_thr - 1), 0, f"{('EDIFFG'):>9}")

        # Plot data points
        for i, yv in enumerate(yvals):
            x, y = map_x(i), map_y(yv)
            canvas[y][x] = char

        title = f"Max Force Convergence (EDIFFG={self.ediffg:.3g} eV/Å, log scale)"
        print("\n" + title)
        put_text(0, max(0, (left_margin - len(y_label)) // 2), y_label)

        for row in canvas:
            print(''.join(row))

        self.log_data["ascii_plot"] = {
            "type": "max_force",
            "scale": "log",
            "width": width,
            "height": height
        }

    def print_report(self, convergence_data=None, problematic=None):
        if convergence_data is None:
            convergence_data = self.calculate_convergence_metrics()
        if problematic is None:
            problematic = self.identify_problematic_atoms()

        print("\n" + "="*80)
        print("VASP Force Convergence Analysis Report")
        print("="*80)

        print("\nConfiguration:")
        print("  EDIFFG: {0:.4f} eV/Å".format(self.ediffg))

        if self.atom_data:
            movable = sum(1 for a in self.atom_data if a.get('movable', True))
            fixed = len(self.atom_data) - movable
            print("  Total atoms: {0}".format(len(self.atom_data)))
            print("  Movable atoms: {0}".format(movable))
            print("  Fixed atoms: {0}".format(fixed))

            element_count = Counter(a.get('element', 'X') for a in self.atom_data)
            print("  Elements: {0}".format(dict(element_count)))

        if convergence_data:
            nsteps = len(convergence_data)
            print("\nOptimization Progress (Total {0} steps):".format(nsteps))
            print("-"*60)
            print("Step     Max Force(eV/Å)    RMS Force(eV/Å)     Status")
            print("-"*60)

            for d in convergence_data:
                status = "Converged" if d['converged'] else "Running"
                print("{0:4d}     {1:12.6f}    {2:12.6f}    {3}".format(
                    d['step'], d['max_force'], d['rms_force'], status))

            print("-"*60)

            final = convergence_data[-1]
            if final['converged']:
                print("\n Optimization converged at step {0}.".format(final['step']))
                print("   Final max force: {0:.6f} eV/Å   (threshold: {1:.4f} eV/Å)".format(
                    final['max_force'], self.ediffg))
            else:
                print("\n Not converged (after {0} steps)".format(nsteps))
                print("   Current max force: {0:.6f} eV/Å   (target: < {1:.4f} eV/Å)".format(
                    final['max_force'], self.ediffg))

        if problematic:
            print("\nProblematic Atoms (Force > {0:.4f} eV/Å):".format(self.ediffg))
            print("-"*80)
            print("Rank  Index  Element  Position                         Force (eV/Å)")
            print("-"*80)

            for rank, atom in enumerate(problematic[:10], 1):
                pos = atom.get('position') or (0.0, 0.0, 0.0)
                print("{0:3d}   {1:4d}   {2:3s}      ({3:8.6f}, {4:8.6f}, {5:8.6f})   {6:10.6f}".format(
                    rank, atom.get('index', -1), atom.get('element', 'X'),
                    pos[0], pos[1], pos[2], atom.get('force_magnitude', 0.0)))

            if len(problematic) > 10:
                print("... {0} more atoms not shown".format(len(problematic) - 10))

    def run(self):
        print("VASP Force Convergence Analyzer")
        print("="*50)
        start_time = time.time()

        self.parse_incar()
        self.parse_poscar()
        self.parse_outcar()

        if not self.force_tables:
            print("No force data found in OUTCAR")
            return

        convergence_data = self.calculate_convergence_metrics()
        problematic = self.identify_problematic_atoms()

        self.print_report(convergence_data=convergence_data, problematic=problematic)
        self.plot_convergence()

        elapsed = time.time() - start_time
        print(f"\nFinished in {elapsed:.2f} s.")


def main():
    analyzer = VASPForceAnalyzer()
    analyzer.run()


if __name__ == "__main__":
    main()