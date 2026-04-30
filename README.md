# KPC-204-test

## download the KPC-2 cristallography
The cristallography is a really good start, we have the exact folding of the protein KPC-2 
```bash
wget https://files.rcsb.org/download/2OV5.pdb
mv 2OV5.pb KPC-2cristalo
```
## Make a KPC-204 .pdb from swiss model.
I don't have cristallography for the KPC-204, i will try to manage with swissmodel in order to have a folding based on the KPC-2 folding and the 3 amino acide include in the KPC-204.
I primarly need a fasta of the protein KPC-204
```bash
wget "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=proteiaan&id=WXU16489.1&rettype=fasta&retmode=text" -O KPC204.fasta
```
On the swiss model web site (https://swissmodel.expasy.org/interactive#sequence) works on the 28/04/2026 add the fasta file and download the .pbd
rename to KCP-204swissmodel.pdb

## Alpha fold
in order to verifie if swiss model made a good folding, I will fold both protein from the scratch.
if the folding of KPC-2 is similare to the cristallography the folding by alphafold will be a good model for KPC-204 otherwise using swissmodel could be the best way.

### downloading micromamba
This micro-environment is a good way to download alpha fold 
```bash 
micromamba create -n colabfold python=3.10 -c conda-forge -y
micromamba activate colabfold
```
and download some tools usefull
```bash 
micromamba install -c bioconda hhsuite -y
pip install pdbfixer
```
### Downloading ColabFold
AlphaFold is thinked for thousand protein and the programme is really heavy and complex, ColabFold is lighter and faster especially if we work on little known protein like i do
```bash
pip install \
    "colabfold[alphafold-minus-jax] @ git+https://github.com/sokrypton/ColabFold"

pip install "jax[cuda12_pip]" \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
I also need to install the weights of the AlphaFold2 neural network.
```python
#python
python -c "from colabfold.download import download_alphafold_params; download_alphafold_params('/data/alexis/colabfold_params')"
```
### launch the prediction 
```bash 
cat > /data/alexis/project/run_colabfold.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=colabfold_KPC204
#SBATCH --output=/data/alexis/project/KPC204/logs/colabfold_%j.log
#SBATCH --error=/data/alexis/project/KPC204/logs/colabfold_%j.err
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

eval "$(micromamba shell hook --shell bash)"
micromamba activate /data/alexis/envs/colabfold

colabfold_batch \
    /data/alexis/project/KPC204.fasta \
    /data/alexis/project/KPC204/output \
    --data /data/alexis/colabfold_params \
    --num-recycle 3 \
    --model-type alphafold2_ptm \
    --templates \
    --amber
EOF

sbatch /data/alexis/project/run_colabfold.sh
```
JAX could not detect GPU if is version is too old, feel free to update it

Copy the file and move it with an easier name
```bash 
cp /data/alexis/project/KPC204/output/WXU16489.1_inhibitor-resistant_carbapenem-hydrolyzing_class_A_beta-lactamase_KPC-204__plasmid___Klebsiella_pneumoniae__relaxed_rank_001_alphafold2_ptm_model_4_seed_000.pdb \
   /data/alexis/project/KPC204/KCPpdb/KPC204alphafold.pdb
```
same for the KPC-2alphafold
```bash
wget -O KPC-2.fasta "https://rest.uniprot.org/uniprotkb/Q9F663.fasta"
```
I have on my repository
```
KPCpdb
|--KPC-2AlphaFold.pdb
|--KPC-2cristalo.pdb
|--KPC-204swissmodel.pdb
|--KPC-204alphafold.pdb
```
I need a comparaison point to validate my KPC proteines's fold to compare i need the tools tmalign
```bash
conda install -c bioconda tmalign
```
In order to compare the structure of protein i use that code 
```bash
TMalign Prot1.pdb Prot2.pdb
```
I have several parameters and I accept a mean deviation of 2 amstrong (RMSD=2) equivalent at a covalent bind and a normality score of 0,70(TMscore=0,70) (Yang Zhang and Jeffrey Skolnick 2004 found that 0,47 was a good score) and an identity of sequence of 0,9 id_seq= 0,9

|Protein 1|Protein 2| TM score | RMSD | seq_ID |word|
|---|---|---|---|---|---|
|KPC-2cristalo_clean.pdb|KCP-2cristalo.pdb|1.000|0.000|1.000|Identical|
|KPC-2cristalo_clean.pdb|KCP-2alphafold.pdb|0.88681|0.42|1.000|really similare protein from alphafold seems to be a good model|
|KPC-204swissmodel.pdb|KCP-204alphafold.pdb|0.89864|1.21|0,978|swiss model looks loke a good model but different from alpha fold|
|KCP-2alphafold.pdb|KCP-204alphafold.pdb|0.94515|1.14|0.993|The comparaison between both alphafold protein looks good the difference probably comes from the 3 amino acide|
|KCP-2alphafold.pdb|KCP-204swissmodel.pdb|0.90273|1.10|0.974|The comparaison between both  protein looks good the difference probably comes from the 3 amino acide|

All protein seems to have really well folded i just decide to exlude KPC-2cristalo.pdb because I have a similar one but cleanner

I can start to modelise with GROMAC
from my .pbd files I will produce a file of coordinate ready for GROMACS
```bash
gmx pdb2gmx -f Protein.pdb -o Protein_processed.gro \
-water tip3p -ff amber99sb-ildn
```
pdb2gmx is a tool from GROMACS that will read the file.pdb add hydrogenes if there are missing, generate the topology (angles, bind, charges) and a forcefield for configure each atomes, creat a file.gro

/!\ my cristalo folding has 2 problems : 
- there is 3 copies of the protein
- there is the ligan inside

I can remove easilly the ligan with grep
```bash
grep -v "BCN" /data/alexis/project/KPC204/KCPpdb/KPC-2cristalo_clean.pdb  > /data/alexis/project/KPC204/KCPpdb/KPC-2cristalo_noligand.pdb
```
and I probably keep only one copy (there are completly indentical each other)

```bash
grep -E "^(ATOM|TER)" /data/alexis/project/KPC204/KCPpdb/KPC-2cristalo_noligand.pdb | awk '$5 == "A"'  > /data/alexis/project/KPC204/KCPpdb/KPC-2cristalo_chainA_only.pdb
```
Next i do the pdb2gmx tool again with the KPC-2cristalo_chainA_only.pdb

I need to introduce the avibactam for the simulation but GROMACS is adapted to protein but not on avibactam, ACPYPE is a molecular tool that translate a molecule to a language that GROMACS understand

```bash
micromamba install -c conda-forge acpype -y
wget "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/CID/25151352/record/SDF/?record_type=3d&response_type=save&response_basename=avibactam" -O avibactam.sdf
acpype -i avibactam.sdf -c bcc -n 0
```
-n0 because the active forme of avibactame is a sulfate zwitterionique, the charge is 0
exit files avibactam_GMX.itp and avibactam_GMX.gro that will interest me 

I need to group them together to make it works for GROMACS
```bash
cat /data/alexis/project/KPC204/KCPpdb/KPCpdb_processed/KCP-2cristallography_processed.gro /data/alexis/avibactam.acpype/avibactam_GMX.gro > /data/alexis/project/KPC_complex/KPC-2cristallography_complex.gro
cat /data/alexis/project/KPC204/KCPpdb/KPCpdb_processed/KCP-204swissmodel_processed.gro /data/alexis/avibactam.acpype/avibactam_GMX.gro > /data/alexis/project/KPC_complex/KPC-204swissmodel_complex.gro
cat /data/alexis/project/KPC204/KCPpdb/KPCpdb_processed/KCP-204alphafold_processed.gro /data/alexis/avibactam.acpype/avibactam_GMX.gro > /data/alexis/project/KPC_complex/KPC-204alphafold_complex.gro
cat /data/alexis/project/KPC204/KCPpdb/KPCpdb_processed/KPC2alphafold_processed.gro /data/alexis/avibactam.acpype/avibactam_GMX.gro > /data/alexis/project/KPC_complex/KPC-2alphafold_complex.gro
```
Here I have a file with both avibactam and KPC protein for each condition.
the problem is that cat kept the nomber of atomes from the protein and did not includ atomes from the avibactam (44)
I have to correct it proprely 
```bash 
# KPC-2 crystallography :3880 +44 = 3924
sed -i '2s/.*/  3924/' /data/alexis/project/KPC_complex/KPC-2cristallography_complex.gro

# KPC-2 alphafold : 4380 + 44 = 4424
sed -i '2s/.*/  4424/' /data/alexis/project/KPC_complex/KPC-2alphafold_complex.gro

# KPC-204 alphafold : 4426 + 44 = 4470
sed -i '2s/.*/  4470/' /data/alexis/project/KPC_complex/KPC-204alphafold_complex.gro

# KPC-204 swissmodel : 4046 + 44 = 4090
sed -i '2s/.*/  4090/' /data/alexis/project/KPC_complex/KPC-204swissmodel_complex.gro
```
Good but it still having a problem here GROMACS looks only the last line and that is the line of the ligan not the protein one

Just need to copy the last line of the proteine coordinate and past it instead of the avibactam one

And because of the cat in putting informations in block I also have 2 line more that I have to remove 
- the informations of the avibactam
- the number of atomes of the avibactamt

  there is a way to identifie them
  ```bash
  for f in *.gro; do
    echo "=== $f ==="
    grep -n "acpype\|avibactam\| 44$\| 44 $" "$f"
    echo ""
done
```

for each .gro that will give you the line of the information of avibactam and the line where there is the number of atomes of avibactam
we can remove it manualy or in case that there is a too much number of files just with this commande

```bash
for f in *.gro; do
    sed -i '/avibactam_GMX.gro created by acpype/d; /^ 44$/d' "$f"
done
```
and the final line to remove is the box vector between the proteine and the avibactam because the cat past in block the proteine.gro again 
i removed it manualy

Ok now I decide to creat the box
in this box we will add 1nm of marge for each side of the proteine to avoid that the proteine interact with itself
there is no wall on the box that means that if the proteine touch the boder a part of it will path to the other side and risque to disturb it.

```bash
gmx editconf -f "$f" \
             -o "${BASE}_box.gro" \
             -c -d 1.0 -bt cubic
```

I notice only one problem that is present from the beggining, the KPC-204alphafold proteine has a XYZ coordinate really elongate on the z axis, that a factor to take and it's means that the folding is probably not optimal.

when I'm looking on the RMSD the higher values are always when KPC-204alphafolds is present, it results will be really criticable at the end

