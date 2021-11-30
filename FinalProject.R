
  ## Pipeline and Methods
  
  The initial R code was taken from [M5 ForecasteR](https://www.kaggle.com/kailex/m5-forecaster) but modified and improved to the point we were able to beat that notebook's score of `0.57`.

### Dataset size issues
The team at first realized the size of the data was going to be an issue. We first examined the data and basic exploratory data analysis and realized the extend and size of the data would most likely be an issue.

First attempts at running on local machines basic models caused errors with memory consumption.

At first we utilize sampling of the data to limit the the number of training samples for training the model, but that achieve horrible performance.  For example, the following used conditional sampling for our "read data" function that was used both in training and prediction.

### XGBoost
We also utilized the R XGBoost library, which provides highly optimized parallel tree boosting algorithms (GBDT, GBM) and allows extensive configuration (https://xgboost.readthedocs.io/en/latest/parameter.html).

create_dt <- function(is_train = TRUE, nrows = Inf, sample_size = 1000) {
  prices <- fread("./data/sell_prices.csv")
  cal <- fread("./data/calendar.csv")
  cal[, `:=`(date = as.IDate(date, format="%Y-%m-%d"),
             is_weekend = as.integer(weekday %chin% c("Saturday", "Sunday")))]
  if (is_train) {
    dt <- fread("./data/sales_train_validation.csv", nrows = nrows)
    dt <- dt[sample(nrow(dt), sample_size), ]
  } else {
    dt <- fread("./data/sales_train_validation.csv", nrows = nrows,
                drop = paste0("d_", 1:(tr_last-max_lags)))
    dt[, paste0("d_", (tr_last+1):(tr_last+2*h)) := NA_real_]
  }
```

Reviewing other examples in Kaggle we ended up taking advantage of the [`data.table`](https://rdatatable.gitlab.io/data.table/) package that provides both fast read capabilities and is memory efficient.  In addition it introduced some operators for the data that helped in providing concise fluent API for querying the data - a "chain like" approach similar to `%>% for other R packages.
The following vignette and article are good overviews:
- https://cran.r-project.org/web/packages/data.table/vignettes/datatable-intro.html
- https://www.machinelearningplus.com/data-manipulation/datatable-in-r-complete-guide/
From the examples the syntax allows complex conditional clauses, SQL like within the statements to make the interaction with the dataset more concise and clearer.
### Dataset for Training
The data provided from Kaggle was essentially "wide" format with a column for every day, and rows representing item sales down to the granular level, but also aggregate records based upon their regional structures.
### Dataset features
The dataset acknowledged holidays, but only on the single observed day. It was our hypothesis that sales trends might be affected for a period prior to the individual events, so we engineered a feature that flagged the 7-day-prior period for each holiday. While this reduced MSE, it did not improve accuracy on the unseen data, implying over-fitting. It's possible that by having week numbers (1-52) is enough give that holidays generally occur at the same time each year
#### Features
The goal was to look at the data more on a weekly vs daily period. Summaries and aggregations for periods were added using a span of 1 week to 4 weeks, in addition to supplementing the periods with holiday information.
The columns were stripped down to just the day columns using a filter based upon a regular expression against the column names - specifically `"^d_"` and this was easily accomplished using the features of `data.table.

Event names and weekend binary were added as well for the day.
The primary features supplementing the data was adding a weekend binary indicator of "is weekend" or not.

Two rolling means were also added - for 7 through 28 days - . These were all look-back periods.

### LightGBM
```
LightGBM is a gradient boosting framework that uses tree based learning algorithms. It is designed to be distributed and efficient with the following advantages:
  ```
- Higher efficiency as well as faster training speed.
- Usage of lower memory.
- Better accuracy.
- Supports Parallel and GPU learning.
- Data of large-scale can be handled


## Acquiring a VERY large VM

At first, memory consumption was an issue. Laptops with 16 Gig or less were not able to handle the entire dataset. Sampling helped, but performance with sampled data wasn't worth it.

Second boundary was CPU. Fortunately, the team has access to Microsoft Azure compute capabilities in for very large Virtual Machines. At one point the team was utilizing a Virtual Machine with 64 cores and 256 Gig of ram. Ultimately, the choice of the underlying package [LightGBM](### LightGBM) the ability to parallel process on many cores permitted a bit better run times.

However, run times were **EXTREMELY LONG** - in some cases 24 hours, and often easily 4 or more hours.

In particular, running DART proved to be a time consuming process (with the runs consistently being longer than 24 hours). In addition, resource monitoring indicated that the DART algorithm was in fact fully utilizing all of the 64 CPUs allocated on the Azure VM for a large part of the cycle.
Via the LightGBM and XGBoost libraries, we investigated the effects of using the following:

- Regression model - linear, Poisson
- Boosting - Gradient Boosting Decision Tree, DART
- Bagging
- Regularization - LASSO, RIDGE, Elastic Network

And the tuning of the following hyper-parameters:

- Learning rate
- Bin size
- Early stop
- Epochs
- Tree depth
- Bagging ratio


```
h <- 28 # forecast horizon
max_lags <- 420 # number of observations to shift by
tr_last <- 1913 # last training day
#...
te <- create_dt(FALSE, nrows)
for (day in as.list(seq(fday, length.out = 2*h, by = "day"))){
  cat(as.character(day), " ")
  tst <- te[date >= day - max_lags & date <= day]
  create_fea(tst)
  tst <- data.matrix(tst[date == day][, c("id", "sales", "date") := NULL])
  te[date == day, sales := predict(m_lgb, tst, n_jobs = cores)]
}