"""Temporal dataset preprocessors for drift experiments.

Provides concrete implementations of temporal data adapters for real-world datasets,
each a BaseTemporalDataset subclass (see src/include/drift_classes.py):
- uber.py: NYC Uber Pickups, spatial coordinates grouped into temporal windows
- wildfire.py: US Wildfire locations, daily temporal aggregation
- household_power.py: UCI Individual Household Electric Power Consumption, daily blocks
- online_retail.py: UCI Online Retail transactions, daily blocks
- twitter.py: UCI Twitter Geospatial Data, per-minute blocks
"""
