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
wget "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=protein&id=WXU16489.1&rettype=fasta&retmode=text" -O KPC204.fasta
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
