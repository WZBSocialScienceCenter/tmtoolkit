set.seed(20200511)
SAMPLE_N <- 1000
OUTPUT_PATH <- '../tmtoolkit/data/'
FILE_PREFIX <- 'parlspeech-v2-sample-'

sample_rows <- function(df, n) {
  rows <- sample(1:nrow(df), n)
  df[rows, ]
}

process_dataset <- function(datainfo) {
  print(sprintf('processing dataset for language %s from %s (label %s)',
                datainfo[1], datainfo[2], datainfo[3]))
  
  df <- readRDS(datainfo[2])
  head(df)
  
  df_sample <- sample_rows(df, SAMPLE_N)
  
  setwd('tmp')
  csvfile <- paste0(datainfo[1], '.csv')
  df_sample$parlspeech_row <- rownames(df_sample)
  write.csv(df_sample, csvfile, row.names = FALSE)
  zip(paste0('../', OUTPUT_PATH, datainfo[1], '/', FILE_PREFIX, datainfo[3], '.zip'), csvfile)
  setwd('..')
  
  print('done')
}

datasets <- list(
  c('de', 'fulldata/Corp_Bundestag_V2.rds', 'bundestag'),
  c('es', 'fulldata/Corp_Congreso_V2.rds', 'congreso'),
  c('en', 'fulldata/Corp_HouseOfCommons_V2.rds', 'houseofcommons'),
  c('nl', 'fulldata/Corp_TweedeKamer_V2.rds', 'tweedekamer')
)

lapply(datasets, process_dataset)

