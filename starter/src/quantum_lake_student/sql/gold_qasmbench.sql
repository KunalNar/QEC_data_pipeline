CREATE SCHEMA IF NOT EXISTS gold;

CREATE TABLE IF NOT EXISTS gold.qasm_circuit (
    circuit_id text PRIMARY KEY,
    source_record_id text NOT NULL UNIQUE,
    benchmark_name text NOT NULL CHECK (benchmark_name <> ''),
    variant text NOT NULL CHECK (variant IN ('source', 'transpiled')),
    member_sha256 text NOT NULL
        CHECK (member_sha256 ~ '^[0-9a-f]{64}$'),
    qubit_count integer NOT NULL CHECK (qubit_count > 0),
    classical_bit_count integer NOT NULL CHECK (classical_bit_count >= 0),
    operation_count integer NOT NULL CHECK (operation_count >= 0),
    measurement_count integer NOT NULL CHECK (measurement_count >= 0),
    two_qubit_gate_count integer NOT NULL
        CHECK (two_qubit_gate_count >= 0
            AND two_qubit_gate_count <= operation_count),
    UNIQUE (benchmark_name, variant)
);

CREATE TABLE IF NOT EXISTS gold.qasm_register (
    circuit_id text NOT NULL REFERENCES gold.qasm_circuit (circuit_id),
    register_name text NOT NULL CHECK (register_name <> ''),
    register_kind text NOT NULL CHECK (register_kind IN ('qreg', 'creg')),
    bit_count integer NOT NULL CHECK (bit_count > 0),
    declaration_order integer NOT NULL CHECK (declaration_order > 0),
    PRIMARY KEY (circuit_id, register_name),
    UNIQUE (circuit_id, declaration_order)
);

CREATE TABLE IF NOT EXISTS gold.qasm_stabilizer_check (
    source_record_id text PRIMARY KEY,
    circuit_id text NOT NULL REFERENCES gold.qasm_circuit (circuit_id),
    check_id text NOT NULL CHECK (check_id <> ''),
    ancilla_qubit text NOT NULL CHECK (ancilla_qubit <> ''),
    syndrome_bit text NOT NULL CHECK (syndrome_bit <> ''),
    UNIQUE (circuit_id, check_id)
);

CREATE TABLE IF NOT EXISTS gold.qasm_check_data_qubit (
    check_source_record_id text NOT NULL
        REFERENCES gold.qasm_stabilizer_check (source_record_id),
    participant_order integer NOT NULL CHECK (participant_order > 0),
    data_qubit text NOT NULL CHECK (data_qubit <> ''),
    PRIMARY KEY (check_source_record_id, participant_order),
    UNIQUE (check_source_record_id, data_qubit)
);

CREATE TABLE IF NOT EXISTS gold.qasm_conditional_correction (
    source_record_id text PRIMARY KEY,
    circuit_id text NOT NULL REFERENCES gold.qasm_circuit (circuit_id),
    condition_register text NOT NULL CHECK (condition_register <> ''),
    condition_value bigint NOT NULL CHECK (condition_value >= 0),
    gate text NOT NULL CHECK (gate <> ''),
    target_qubit text NOT NULL CHECK (target_qubit <> ''),
    FOREIGN KEY (circuit_id, condition_register)
        REFERENCES gold.qasm_register (circuit_id, register_name)
);

CREATE INDEX IF NOT EXISTS qasm_check_circuit_syndrome_idx
    ON gold.qasm_stabilizer_check (circuit_id, syndrome_bit);

CREATE INDEX IF NOT EXISTS qasm_correction_circuit_condition_idx
    ON gold.qasm_conditional_correction
    (circuit_id, condition_register, condition_value);
