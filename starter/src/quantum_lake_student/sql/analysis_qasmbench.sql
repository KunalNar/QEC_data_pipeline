-- Repetition-code parity checks and syndrome-controlled recovery.
-- The integer condition uses bit i for syndrome register bit [i].
SELECT
    c.benchmark_name,
    c.variant,
    k.ancilla_qubit,
    participants.data_qubits,
    k.syndrome_bit,
    r.condition_value,
    r.gate AS correction_gate,
    r.target_qubit AS correction_target
FROM gold.qasm_circuit AS c
JOIN gold.qasm_stabilizer_check AS k USING (circuit_id)
JOIN LATERAL (
    SELECT array_agg(p.data_qubit ORDER BY p.participant_order) AS data_qubits
    FROM gold.qasm_check_data_qubit AS p
    WHERE p.check_source_record_id = k.source_record_id
) AS participants ON true
JOIN gold.qasm_conditional_correction AS r
    ON r.circuit_id = c.circuit_id
    AND r.condition_register = split_part(k.syndrome_bit, '[', 1)
    AND (
        (r.condition_value >> split_part(
            split_part(k.syndrome_bit, '[', 2), ']', 1
        )::integer) & 1
    ) = 1
WHERE c.benchmark_name = 'qec_sm_n5'
ORDER BY c.variant, k.syndrome_bit, r.condition_value;
