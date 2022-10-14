# axonal_analysis.py
used for analyzing a folder of rgb .tif images for various parameters related to axonal segments of peripheral nerve


[Mokarram N, Dymanus K, Srinivasan A, Lyon JG, Tipton J, Chu J, English AW, Bellamkonda RV. Immunoengineering nerve repair. Proc Natl Acad Sci U S A. 2017 Jun 27;114(26):E5077-E5084. doi: 10.1073/pnas.1705757114. Epub 2017 Jun 13. PMID: 28611218](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5495274/)

The images will be interpreted thusly: 
    GREEN: axons 
    RED: myelin
    BLUE: nuclei
    
The two primary forms of measurement are: 
    1) The overall coverage of a given color above a specified threshold
    2) The thickness of myelin around each axon
    
Axons are first detected as particles and then using the individual centers, a vector is 
swept from that center to measure any myelination with a specified proximity of the axon's
edge. Using these vectors, axons are grouped as either myelinated or unmyelinated based on 
a specified percentage of valid vectors per axon. For myelinated axons, myelin thickness 
is calculated and reported as an average per axon.
