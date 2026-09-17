# Data Directory

This project does not store downloaded seismic data in Git.

## Directory policy

```text
data/
├── manifests/     # Version-controlled source, licensing, and provenance notes
├── raw/           # Downloaded SEG-Y files; ignored by Git
├── interim/       # Temporary conversion and inspection outputs; ignored by Git
└── processed/     # Derived arrays and scores; ignored by Git
```

## First real-data source

The initial field-data case study uses the USGS Cape Cod Bay 2019-002-FA
multichannel seismic-reflection release.

- Landing page: https://cmgds.marine.usgs.gov/data-releases/datarelease/10.5066-P99DR4PN/
- DOI: https://doi.org/10.5066/P99DR4PN
- Manifest: `manifests/usgs_2019_002_fa_mcs.yaml`

The release includes multichannel SEG-Y trace data, navigation, CMP products,
and profile images. The project uses the data only for an unsupervised real-data
QC case study unless independent trace-defect labels are created and documented.

## Attribution

Retain USGS attribution and the DOI in all derived documents and figures.

## Do not commit raw data

Do not commit:

- `.sgy` or `.segy` files
- downloaded archives
- large processed arrays
- raw navigation exports
- machine-specific output artifacts

Commit only manifests, source code, small permitted derived figures, and
reproducible download/inspection instructions.