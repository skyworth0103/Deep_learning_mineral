# Order to plot
ORDER = [16, 17, 14, 18, 15, 20, 21, 24, 23, 26, 27, 28, 29, 25, 6, 7, 5, 3, 4,
         9, 10, 2, 8, 11, 22, 19, 12, 13, 0, 1]
# Isolate label
STRAINS = {0: "C. albicans", 1: "C. glabrata", 2: "K. aerogenes", 3: "E. coli 1", 4: "E. coli 2",
           5: "E. faecium",  6: "E. faecalis 1", 7: "E. faecalis 2", 8: "E. cloacae",
           9: "K. pneumoniae 1", 10: "K. pneumoniae 2", 11: "P. mirabilis", 12: "P. aeruginosa 1",
           13: "P. aeruginosa 2", 14: "MSSA 1", 15: "MSSA 3", 16: "MRSA 1",
           17: "MRSA 2", 18: "MSSA 2", 19: "S. enterica", 20: "S. epidermidis", 21: "S. lugdunensis",
           22: "S. marcescens", 23: "S. pneumoniae 2", 24: "S. pneumoniae 1", 25: "S. sanguinis",
           26: "Group A Strep.", 27: "Group B Strep.", 28: "Group C Strep.", 29: "Group G Strep."}

# Assigin isolate to treatment Groups
ATCC_GROUPINGS = {3: 0,
                  4: 0,
                  9: 0,
                  10: 0,
                  2: 0,
                  8: 0,
                  11: 0,
                  22: 0,
                  12: 2,
                  13: 2,
                  14: 3,  # MSSA
                  18: 3,  # MSSA
                  15: 3,  # MSSA
                  20: 3,
                  21: 3,
                  16: 3,  # isogenic MRSA
                  17: 3,  # MRSA
                  23: 4,
                  24: 4,
                  26: 5,
                  27: 5,
                  28: 5,
                  29: 5,
                  25: 5,
                  6: 5,
                  7: 5,
                  5: 6,
                  19: 1,
                  0: 7,
                  1: 7}


ab_order = [3, 4, 5, 6, 0, 1, 2, 7]
antibiotics = {0: "Meropenem", 1: "Ciprofloxacin", 2: "TZP",
               3: "Vancomycin", 4: "Ceftriaxone", 5: "Penicillin",
               6: "Daptomycin", 7: "Caspofungin"}
