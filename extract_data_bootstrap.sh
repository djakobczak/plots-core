#!/bin/bash

set -u

TYPE="${1:-"vm"}"
TEST_DIR="${2}"

__remove_color(){
    # sed -r "s/\x1B\[(([0-9]{1,2})?(;)?([0-9]{1,2})?)?[m,K,H,f,J]//g" $1
    # sed -r "s/\x1B\[([0-9]{1,3}(;[0-9]{1,2})?)?[mGK]//g" $1
    sed "s,\x1B\[[0-9;]*[a-zA-Z],,g"  $1
}

RESULT_FILE="test-bootstrap-${TYPE}-results.csv"
TEST_STARTED=$(cat ${TEST_DIR}/general.log | grep -i "test start" | tr -s ' ' | cut -d ' ' -f 1 | cut -c -15)
UPLANCE_ALIVE="$(cat ${TEST_DIR}/uplane_alive.log | grep -i "ping" | tr -s ' ' | cut -d ' ' -f 1 | cut -c -15)"
CPLANCE_ALIVE="$(cat ${TEST_DIR}/cplane_alive.log | grep -i "ping" | tr -s ' ' | cut -d ' ' -f 1 | cut -c -15)"
if [[ "${TYPE}" == "vm" ]]; then
    PFCP_ASSOCIATED="$(cat ${TEST_DIR}/open5gs/upf.log | tail | grep -i "pfcp associated" | tr -s ' ' |  cut -d ' ' -f 2 | rev | cut -c 2- | rev | cut -c -15)"
    GNB_SESSION="$(grep -i "NG Setup procedure is successful" ${TEST_DIR}/gnb.log | tr -s ' ' | cut -d ' ' -f 2 | tr -d ']' | cut -c -15)"
elif [[ "${TYPE}" == "container" ]]; then
    PFCP_ASSOCIATED="$(__remove_color ${TEST_DIR}/open5gs.log | grep "upf" | grep -i "has already been associated" | tr -s ' ' |  cut -d ' ' -f 4 | rev | cut -c 2- | rev)"
    GNB_SESSION="$(__remove_color ${TEST_DIR}/gnb.log | grep -i "NG Setup procedure is successful" | tr -s ' ' | cut -d ' ' -f 4 | tr -d ']' | cut -c -15)"
fi

if [[ ! -f "${RESULT_FILE}" ]]; then
    echo "test_type,start_test,uplane_ping,cplane_ping,pfcp_associated,gnb_session" > ${RESULT_FILE}
fi
echo "${TYPE},${TEST_STARTED},${UPLANCE_ALIVE},${CPLANCE_ALIVE},${PFCP_ASSOCIATED},${GNB_SESSION}" >> ${RESULT_FILE}
