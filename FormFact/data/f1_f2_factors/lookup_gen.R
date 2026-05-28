library(stringr)
library(tidyverse)
library(jsonlite)

# load all TSV files 
tsv_paths <- list.files(path = , pattern = "\\.tsv$", full.names = TRUE)
# initialize empty df
df_cache <- NULL 

# iterate through, load df 
for (tsv in tsv_paths) {
    # load df
    df <- read_tsv(tsv)
    
    # strip element name
    element_name <- str_remove(names(df)[2], "f[12]_")
    
	# round E_ev to nearest thousandth
	df <- df %>% mutate(E_ev = round(E_ev, 3))

    # combine f1_<element_name> f2_<element_name> into 
    # a single column named <element_name> w/ list for f1,f2
    df <- df %>% 
        rowwise() %>%
        mutate(to_change = list(c(.data[[names(df)[2]]], .data[[names(df)[3]]]))) %>%
        ungroup() %>%
        rename(!!element_name := to_change) %>%
        select(-starts_with("f"))
    
    # Combine dataframes
    if (is.null(df_cache)) {
        df_cache <- df
    } else {
        df_cache <- left_join(df_cache, df, by = "E_ev")    
    }
}

# Convert to nested structure: energy -> {element: [f1, f2]}
result <- df_cache %>%
    group_split(E_ev) %>%
    map(~{
        energy <- .x$E_ev[1]
        elements <- .x %>% select(-E_ev) %>% as.list()
        list(energy = energy, data = elements)
    }) %>%
    setNames(map_chr(., ~as.character(.x$energy))) %>%
    map(~.x$data)

# Write to file
write_json(result, "factor_lookup.json", pretty = TRUE, digits = NA, auto_unbox = FALSE)