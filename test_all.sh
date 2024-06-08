# go to self folder
# DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# cd $DIR

# check cuda available
./check_cuda.py 2>/dev/null
if [[ $? -ne 0 ]]; then
    echo "CUDA check failed"
    exit 1
fi

# do tests
./test_matrix.py 64
./test_matrix.py 32
./test_matrix.py 16
./test_matrix.py b16
./train_all.sh
./inference_all.sh