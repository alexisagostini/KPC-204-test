# KPC-204-FINAL
# KPC β-lactamase + Avibactam — Molecular Dynamics Simulations

> Comparative MD study of KPC-2 (wild-type) and KPC-204 (V204 variant) β-lactamases  
> in complex with avibactam, using crystallographic and homology-modeled structures.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Dependencies](#dependencies)
- [Current Status](#CURRENT-STATUS)
- [Pipeline summary](#PIPELINE-SUMMARY)
- [Full PIPELINE with commande](#Full-PIPELINE-with-commande)
- [Analysis already done](#ANALYSIS-ALREADY-DONE-(deprecated-runs-—-wrong-ligand))
- [Immediate priorities](#IMMEDIATE-PRIORITIES)


## Project Overview

This project investigates the binding dynamics of **avibactam** (a non-β-lactam β-lactamase inhibitor)  
within the active site of two KPC variants:

| Protein | Structure Source | Ligand |
|---------|-----------------|--------|
| KPC-2 | X-ray crystallography | Avibactam (CID 9835049) |
| KPC-204 (V204 variant) | SwissModel homology model | Avibactam (CID 9835049) |

Each system underwent 100 ns of classical MD simulation using **GROMACS 2025.4**  
with the **AMBER99SB-ILDN** force field and **GAFF2** parameters for the ligand.


## Repository Structure
```
├── docking/
│   ├── avibactam_REAL.pdbqt          # Correct avibactam ligand (CID 9835049)
│   ├── KPC2_cristallo/
│   │   ├── KPC2_cristallo_receptor.pdbqt
│   │   ├── KPC2_cristallo_docked.pdbqt
│   │   ├── KPC2_cristallo_best_pose.pdb
│   │   └── vina_config.txt
│   └── KPC204_swissmodel/
│       ├── KPC204_swissmodel_receptor.pdbqt
│       ├── KPC204_swissmodel_docked.pdbqt
│       ├── KPC204_swissmodel_best_pose.pdb
│       └── vina_config.txt
│
├── ligand/
│   └── avibactam_REAL.acpype/
│       ├── avibactam_REAL_GMX.itp    # GAFF2 topology
│       ├── avibactam_REAL_GMX.gro    # Ligand coordinates
│       └── posre_avibactam_REAL.itp  # Position restraints
│
├── mdp/
│   ├── ions.mdp                      # Minimal MDP for genion
│   ├── em.mdp                        # Energy minimization
│   ├── nvt.mdp                       # NVT equilibration (300K, 100 ps)
│   ├── npt.mdp                       # NPT equilibration (1 bar, 100 ps)
│   └── md.mdp                        # Production MD (100 ns)
│
├── systems/
│   ├── KPC2_cristallo_v2/            # ← ACTIVE (correct ligand)
│   │   ├── protein_clean.pdb
│   │   ├── MOL.itp
│   │   ├── topol.top
│   │   ├── complex_ions.gro
│   │   ├── em.gro / nvt.gro / npt.gro
│   │   └── md.xtc / md.tpr
│   └── KPC204_swissmodel_v2/         # ← ACTIVE (correct ligand)
│       ├── protein_clean.pdb
│       ├── MOL.itp
│       ├── topol.top
│       └── ...
│
└── analysis/
├── plot_analysis.py              # RMSD / RMSF / Rg plots
├── rmsd.xvg
├── rmsf.xvg
├── gyrate.xvg
└── dist_ser70_C7.xvg             # Ser70–Avibactam C7 distance
# help from claude for the format beyong
```

## Dependencies

| Tool | Version | Purpose |
|------|---------|---------|
| GROMACS | 2025.4 | MD engine |
| ACPYPE | 2023.10.27 | GAFF2 ligand parameterization |
| AutoDock Vina | 1.x | Molecular docking |
| Open Babel | 3.1.0 | Format conversion |
| Python | ≥ 3.10 | Analysis scripts |
| matplotlib / numpy | latest | Plotting |

## CURRENT STATUS

### KPC2_cristallo_v2
- Production MD running on GPU 0 (PID 1857373)
- 100 ns simulation, AMBER99SB-ILDN + GAFF2, TIP3P water, 0.15 M NaCl

### KPC204_swissmodel_v2
- Energy minimization (EM) blocked — gmx mdrun hangs silently after printing the GROMACS header, produces no em.log
- em.tpr is valid (confirmed via gmx dump: steep integrator, 10000 steps, emtol=100)
- No file locks, no backup conflicts
- Tested: GPU and CPU-only modes both hang at the same point
- Root cause: unknown — suspected CUDA context conflict or MPI detection hang
## CRITICAL NOTES — READ CAREFULLY

### WRONG LIGAND IN OLD RUNS
All systems WITHOUT _v2 suffix used PubChem CID 25151352 (a fluorochlorinated compound: Cl, 3×F, no S, no O) instead of real avibactam. DO NOT use these for analysis.

### TOPOLOGY FIX REQUIRED
ACPYPE places [ atomtypes ] inside MOL.itp. GROMACS requires it in topol.top immediately after the forcefield include. Fix:
  1. Extract lines 3–18 of MOL.itp (the [ atomtypes ] block)
  2. Inject into topol.top after the forcefield.itp include line
  3. Remove those lines from MOL.itp (keep from [ moleculetype ] onward)
  4. Rename molecule: sed -i 's/avibactam_REAL/MOL/g' MOL.itp

### NON-COVALENT SIMULATION
Avibactam forms a covalent adduct with Ser70 in reality. These are classical non-covalent simulations. The Ser70(OG)–Avibactam(C7) distance reflects pre-covalent binding competence only.

### ATOM NUMBERS PER SYSTEM
Atom numbers for Ser70 OG and MOL C7 differ between systems. Always extract from frame0.pdb:
  grep " OG  SER A  70" frame0.pdb
  grep " C7  MOL"        frame0.pdb

Known values (frame 0):
  KPC2_cristallo:    OG=573,  C7=3896
  KPC2_alphafold:    OG=1046, C7=4396
  KPC204_alphafold:  OG=1046, C7=4442
  KPC204_swissmodel: OG=666,  C7=4063

## PIPELINE SUMMARY

Step 1 — Ligand: acpype -i avibactam_REAL.sdf -c bcc -a gaff2 -o gmx -n 0
Step 2 — Docking: AutoDock Vina with existing receptor.pdbqt files
Step 3 — Protein topology: gmx pdb2gmx -ff amber99sb-ildn -water spce -ignh
Step 4 — Complex: merge protein.gro + MOL.gro, update atom count in line 2
Step 5 — Solvation: gmx editconf (dodecahedron, 1.2 nm) → gmx solvate
Step 6 — Ions: gmx grompp + gmx genion (-neutral -conc 0.15)
Step 7 — EM: gmx grompp + gmx mdrun -ntmpi 1 -ntomp 8 -gpu_id X -nb gpu
Step 8 — NVT: 300K, 100 ps, position restraints
Step 9 — NPT: 1 bar, 100 ps, position restraints
Step 10 — MD: 100 ns production

## Full PIPELINE with commande
### STEP 0 — Environment setup
```
conda activate gmx_gpu

ROOT=/data/alexis/project/grmcomplex
BASE=$ROOT/systems
MDP=$ROOT/mdp
LIG=$ROOT/ligand/avibactam_REAL.acpype
DOCK=$ROOT/docking
```
### STEP 1 — Ligand preparation (ACPYPE)

cd $ROOT/ligand

# Download correct avibactam (CID 9835049)
```
wget "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/CID/9835049/SDF?record_type=3d" \
     -O avibactam_REAL.sdf
```
# Verify: must have S and O, no Cl or F ( moluecule used for the crystallography )
```
grep " S " avibactam_REAL.sdf   # must return 1 line
grep " F " avibactam_REAL.sdf   # must return nothing
grep " Cl" avibactam_REAL.sdf   # must return nothing
```
# Generate GAFF2 topology (Litlle organic molecule like avibactam)
```
acpype -i avibactam_REAL.sdf -c bcc -a gaff2 -o gmx -n 0

# Rename molecule to MOL (required for GROMACS compatibility)
sed -i 's/avibactam_REAL/MOL/g' $LIG/avibactam_REAL_GMX.itp
```
Expected output in $LIG/:
avibactam_REAL_GMX.itp  → GAFF2 topology (28 atoms: S,O×6,N×3,C×7,H×11)
avibactam_REAL_GMX.gro  → ligand coordinates
posre_avibactam_REAL.itp

### STEP 2 — Molecular docking (AutoDock Vina)
```
cd $DOCK
obabel $ROOT/ligand/avibactam_REAL.sdf -O avibactam_REAL.pdbqt \
       --partialcharge gasteiger -h
sed -i "s|avibactam.pdbqt|avibactam_REAL.pdbqt|g" \
    KPC2_cristallo/vina_config.txt KPC204_swissmodel/vina_config.txt
vina --config KPC2_cristallo/vina_config.txt
vina --config KPC204_swissmodel/vina_config.txt

# Extract best pose
for sys in KPC2_cristallo KPC204_swissmodel; do
    python3 -c "
lines = open('${sys}/${sys}_docked.pdbqt').readlines()
out, in_m = [], False
for l in lines:
    if l.strip() == 'MODEL 1': in_m = True
    if in_m: out.append(l)
    if in_m and l.startswith('ENDMDL'): break
open('${sys}/${sys}_best_pose.pdbqt','w').writelines(out)
"
    obabel $DOCK/${sys}/${sys}_best_pose.pdbqt \
           -O $DOCK/${sys}/${sys}_best_pose.pdb 2>/dev/null
done
```
### STEP 3 — Create system directories and copy files
```
for sys in KPC2_cristallo KPC204_swissmodel; do
    mkdir -p $BASE/${sys}_v2
    cp $BASE/${sys}/protein_clean.pdb          $BASE/${sys}_v2/
    cp $DOCK/${sys}/${sys}_best_pose.pdb       $BASE/${sys}_v2/ligand.pdb
    cp $LIG/avibactam_REAL_GMX.itp             $BASE/${sys}_v2/MOL.itp
    cp $LIG/avibactam_REAL_GMX.gro             $BASE/${sys}_v2/MOL.gro
    cp $LIG/posre_avibactam_REAL.itp           $BASE/${sys}_v2/posre_MOL.itp
    sed -i 's/avibactam_REAL/MOL/g'            $BASE/${sys}_v2/MOL.itp
    echo "Ready: ${sys}_v2"
done
```
### STEP 4 — Protein topology (pdb2gmx)
```
for sys in KPC2_cristallo KPC204_swissmodel; do
    cd $BASE/${sys}_v2
    gmx pdb2gmx -f protein_clean.pdb -o protein.gro -p topol.top \
                -water spce -ff amber99sb-ildn -ignh
done
```
### STEP 5 — Build complex GRO (protein + ligand)
```
for sys in KPC2_cristallo KPC204_swissmodel; do
    cd $BASE/${sys}_v2
    NATOM_PROT=$(sed -n '2p' protein.gro | tr -d ' ')
    NATOM_LIG=$(sed -n '2p' MOL.gro | tr -d ' ')
    NATOM_TOTAL=$((NATOM_PROT + NATOM_LIG))
    head -1 protein.gro > complex.gro
    echo " $NATOM_TOTAL" >> complex.gro
    sed -n '3,$p' protein.gro | head -n $NATOM_PROT >> complex.gro
    sed -n '3,$p' MOL.gro     | head -n $NATOM_LIG  >> complex.gro
    tail -1 protein.gro >> complex.gro
done
```
### STEP 6 — Fix topol.top (CRITICAL — atomtypes placement)

ACPYPE puts [ atomtypes ] inside MOL.itp — GROMACS needs it in topol.top.
This step moves it to the right place.
```
for sys in KPC2_cristallo KPC204_swissmodel; do
    cd $BASE/${sys}_v2
    cp MOL.itp MOL.itp.bak && cp topol.top topol.top.bak

    MOLTYPE_LINE=$(grep -n "^\[ moleculetype \]" MOL.itp | head -1 | cut -d: -f1)
    sed -n "3,$((MOLTYPE_LINE - 1))p" MOL.itp > atomtypes_block.tmp
    sed -n "${MOLTYPE_LINE},\$p" MOL.itp > MOL_clean.itp && mv MOL_clean.itp MOL.itp

    FFLINE=$(grep -n "forcefield.itp" topol.top | head -1 | cut -d: -f1)
    sed -i "${FFLINE}r atomtypes_block.tmp" topol.top

    SYSLINE=$(grep -n "^\[ system \]" topol.top | head -1 | cut -d: -f1)
    sed -i "$((SYSLINE - 1))a #include \"MOL.itp\"\n#include \"posre_MOL.itp\"" topol.top
    echo "MOL                  1" >> topol.top
done
```
### STEP 7 — Solvate
```
for sys in KPC2_cristallo KPC204_swissmodel; do
    cd $BASE/${sys}_v2
    gmx editconf -f complex.gro -o complex_box.gro -c -d 1.2 -bt dodecahedron
    gmx solvate  -cp complex_box.gro -cs spc216.gro -o complex_solv.gro -p topol.top
done
```
### STEP 8 — Add ions
```
for sys in KPC2_cristallo KPC204_swissmodel; do
    cd $BASE/${sys}_v2
    gmx grompp -f $MDP/ions.mdp -c complex_solv.gro -r complex_solv.gro \
               -p topol.top -o ions.tpr -maxwarn 2
    echo "SOL" | gmx genion -s ions.tpr -o complex_ions.gro \
                 -p topol.top -pname NA -nname CL -neutral -conc 0.15
done
```
### STEP 9 — Energy minimization
```
for sys in KPC2_cristallo KPC204_swissmodel; do
    cd $BASE/${sys}_v2
    gmx grompp -f $MDP/em.mdp -c complex_ions.gro -r complex_ions.gro \
               -p topol.top -o em.tpr -maxwarn 2
    gmx mdrun -v -deffnm em -ntmpi 1 -ntomp 8 -gpu_id 1 -nb gpu
done
 ```
If it hangs → use CPU instead:
gmx mdrun -v -deffnm em -ntmpi 1 -ntomp 8 -nb cpu -pme cpu -bonded cpu

### STEP 10 — NVT (300 K, 100 ps)
```
for sys in KPC2_cristallo KPC204_swissmodel; do
    cd $BASE/${sys}_v2
    gmx grompp -f $MDP/nvt.mdp -c em.gro -r em.gro \
               -p topol.top -o nvt.tpr -maxwarn 2
    gmx mdrun -v -deffnm nvt -ntmpi 1 -ntomp 8 -gpu_id 1
done
```
### STEP 11 — NPT (1 bar, 100 ps)
```
for sys in KPC2_cristallo KPC204_swissmodel; do
    cd $BASE/${sys}_v2
    gmx grompp -f $MDP/npt.mdp -c nvt.gro -r nvt.gro -t nvt.cpt \
               -p topol.top -o npt.tpr -maxwarn 2
    gmx mdrun -v -deffnm npt -ntmpi 1 -ntomp 8 -gpu_id 1
done
```
### STEP 12 — Production MD (100 ns)
```
screen -S md_v2   # detach with Ctrl+A then D — NEVER Ctrl+Z that made me pass lot of time to understand what happen ! 

for sys in KPC2_cristallo KPC204_swissmodel; do
    cd $BASE/${sys}_v2
    gmx grompp -f $MDP/md.mdp -c npt.gro -t npt.cpt \
               -p topol.top -o md.tpr -maxwarn 2
    gmx mdrun -v -deffnm md -ntmpi 1 -ntomp 8 -gpu_id 1 \
              -nb gpu -pme gpu -bonded gpu
done
```
### STEP 13 — Analysis (after MD) with a big claude help to fix the code! 
```
for sys in KPC2_cristallo KPC204_swissmodel; do
    cd $BASE/${sys}_v2

    # Recenter trajectory
    echo "Protein System" | gmx trjconv -s md.tpr -f md.xtc \
        -o md_center.xtc -center -pbc mol -ur compact

    # RMSD
    echo "Backbone Backbone" | gmx rms -s md.tpr -f md_center.xtc \
        -o rmsd.xvg -tu ns

    # RMSF
    echo "Backbone" | gmx rmsf -s md.tpr -f md_center.xtc \
        -o rmsf.xvg -res

    # Ser70–Avibactam C7 distance
    echo "System" | gmx trjconv -s md.tpr -f md_center.xtc \
        -o frame0.pdb -dump 0 2>/dev/null
    OG=$(grep " OG  SER A  70" frame0.pdb | awk '{print $2}')
    C7=$(grep " C7  MOL"        frame0.pdb | awk '{print $2}')
    printf "[ OG_Ser70 ]\n${OG}\n[ C7_MOL ]\n${C7}\n" > dist_ser70.ndx
    gmx distance -s md.tpr -f md_center.xtc -n dist_ser70.ndx \
        -select 'com of group "OG_Ser70" plus com of group "C7_MOL"' \
        -oall dist_ser70_C7.xvg -tu ns
done
```
## ANALYSIS ALREADY DONE (deprecated runs — wrong ligand)

- md_center.xtc generated for all 4 old systems
- RMSD, RMSF, gyrate computed
- Ser70–C7 distances computed:
    KPC2_cristallo:    1.258 ± 0.423 nm
    KPC2_alphafold:    1.685 ± 0.254 nm
    KPC204_alphafold:  2.298 ± 0.484 nm
    KPC204_swissmodel: 1.088 ± 0.047 nm
- Comparative plots saved in systems/ as PNG files

## IMMEDIATE PRIORITIES

1. Fix KPC204_swissmodel_v2 EM blocking issue
