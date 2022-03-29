#!/bin/bash
set -u

FN="${1}"

# check if number of headers are same as row cells, if so return
N_EXTRA_COMMAS="$(grep -io "%cpu" $FN | wc -l)"  # float with commas
N_TARGET_CELLS=$(tail -n 10 $FN | awk -F',' '{print NF; exit}')
N_HEADER_CELLS=$(awk -F',' '{print NF; exit}' $FN)
let N_TARGET_CELLS=${N_TARGET_CELLS}-${N_EXTRA_COMMAS}
if [[ "$N_TARGET_CELLS" == "$N_HEADER_CELLS" ]]; then
    echo "File already prepared, exit..."
    exit 0
fi

let MISSING_NFS="(${N_TARGET_CELLS}-${N_HEADER_CELLS})/11"  # 10 headers + 1 extra comma
echo "Missing headers: $MISSING_NFS"
HEADERS=",Domain ID,Domain name,CPU (ns),%CPU,Mem (bytes),%Mem,Block RDRQ,Block WRRQ,Net RXBY,Net TXBY"
for i in $(seq 1 $MISSING_NFS); do
    sed -i "1s/$/${HEADERS}/"  $FN
done
