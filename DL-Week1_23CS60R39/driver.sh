#!/bin/bash

#chmod +x driver.sh
#./driver.sh
python_programs=(
    "preprocess.py"
    "CNN.py"
    "DNN.py"
    "LSTM.py"
    "evaluate.py"
)

# Loop through each program and run it
for program in "${python_programs[@]}"; do
    echo "Running $program..."
    python3 "$program"
    echo "$program finished."
    echo
done

echo "All programs executed."