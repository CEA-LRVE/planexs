"""Surrogate-model construction. The search-space encoding it works on lives in
`planexs.encoding` (roadmap task P3-01). A single, swappable factory function
(`buildSurrogateModel`) so a sparse/scalable GP can be plugged in later (section 4.4)
without touching the acquisition or API layers.
"""
