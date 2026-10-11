"""Surrogate-model construction and the search-space encoding it works on. A
single, swappable factory function (`buildSurrogateModel`) so a sparse/scalable
GP can be plugged in later (section 4.4) without touching the acquisition or
API layers.
"""
