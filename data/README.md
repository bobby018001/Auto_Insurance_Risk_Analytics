# Data placement

The original course archive includes anonymized training and test tables, but it does not include a public redistribution license. They are intentionally excluded from the public repository.

Authorized users should place files as follows:

```text
data/
├── raw/
│   ├── insurance_train2026.csv
│   └── insurance_test2026.csv
└── processed/
    ├── train_X.csv
    ├── test_X.csv
    ├── train_y.csv
    └── clip_bounds.csv
```

## Original dimensions

- Training data: 39,928 rows and 28 columns
- Test data: 13,310 rows and 23 columns
- Raw columns use anonymized labels such as `X.1` through `X.28`

The cleaning notebook generates the processed files. Target-related fields must remain excluded from feature matrices to prevent leakage.
