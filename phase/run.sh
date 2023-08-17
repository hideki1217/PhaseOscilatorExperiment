#!/bin/bash

SCRIPT_DIR=$(cd $(dirname $0); pwd)
cd ${SCRIPT_DIR}

# Run main routin
echo $(time ./build/phase ${SCRIPT_DIR}) > ${SCRIPT_DIR}/build/result.log
# Analyze the result of main routin
echo $(time python3 ./scripts/phase.py ${SCRIPT_DIR}) >> ${SCRIPT_DIR}/build/result.log
# Notify the end of the task
python3 ../notify.py ${SCRIPT_DIR}/build/result.log
