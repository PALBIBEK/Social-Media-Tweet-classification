#!/bin/bash

#chmod +x driver.sh
#./driver.sh
python_programs=(
    "Preproces.py"
    "Vectorize.py"
    "CNN.py"
    "DNN.py"
    "LSTM.py"
    "RunEval.py"
)

# Loop through each program and run it
for program in "${python_programs[@]}"; do
    echo "Running $program..."
    python3 "$program"
    echo "$program finished."
    echo
done

echo "All programs executed."
