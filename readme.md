1. kita download pip install requiremnet yang ada pada document 

2. kita bentuk src document processor dimana kita harus download model sentence-transformers/all-MiniLM-L6-v2" 
# Make sure git-lfs is installed (https://git-lfs.com)
git lfs install

git clone https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

# If you want to clone without large files - just their pointers
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

# Make sure huggingface-cli is installed: pip install -U "huggingface_hub[cli]"
huggingface-cli download sentence-transformers/all-MiniLM-L6-v2


3. download deepseek 1.3b base from hugging face and git clone lfs -> git lfs pull 
