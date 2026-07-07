-- Table principale : séjours patients nettoyés pour la prédiction de réhospitalisation
CREATE TABLE IF NOT EXISTS patient_encounters (
    encounter_id            BIGINT PRIMARY KEY,
    patient_nbr             BIGINT NOT NULL,

    -- Démographie
    race                    VARCHAR(30),
    gender                  VARCHAR(10),
    age                     VARCHAR(10),

    -- Admission / séjour
    admission_type_id       SMALLINT,
    discharge_disposition_id SMALLINT,
    admission_source_id     SMALLINT,
    time_in_hospital        SMALLINT,
    medical_specialty       VARCHAR(50),

    -- Procédures et examens
    num_lab_procedures      SMALLINT,
    num_procedures          SMALLINT,
    num_medications         SMALLINT,

    -- Historique hospitalier (antécédents)
    number_outpatient       SMALLINT,
    number_emergency        SMALLINT,
    number_inpatient        SMALLINT,

    -- Diagnostics
    diag_1                  VARCHAR(10),
    diag_2                  VARCHAR(10),
    diag_3                  VARCHAR(10),
    number_diagnoses        SMALLINT,

    -- Résultats biologiques
    max_glu_serum           VARCHAR(20),
    a1c_result              VARCHAR(20),

    -- Traitement
    diabetes_med            VARCHAR(5),
    change_med               VARCHAR(5),

    -- Variable cible
    readmitted              VARCHAR(5),
    target                   SMALLINT NOT NULL,

    created_at              TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index utiles pour les requêtes fréquentes
CREATE INDEX IF NOT EXISTS idx_patient_nbr ON patient_encounters(patient_nbr);
CREATE INDEX IF NOT EXISTS idx_target ON patient_encounters(target);