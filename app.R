# app.R -- Shiny app for bivariate analysis, Y vs X
# Built from Bivariate_and_OLS.R (step 1) and Data_for_thisclass_noNAs.dta
# transformed to R as data_for_class.csv in this folder.
#
# How to run:
#   setwd() to this folder, then in R:
#     shiny::runApp()
#   or from a terminal in this folder:
#     Rscript -e "shiny::runApp()"
#   Packages needed: shiny, data.table, ggplot2.
#   Install them once with:
#     install.packages(c("shiny", "data.table", "ggplot2"))
#
# Layout (as specified):
#   Top: Y and X selectors. Y is restricted to phq9, bmi,
#     depression and pov_inc_ratio; X can be any variable except
#     less_9, nine_11, hs_ged, scoll and coll.
#   Row 1, column 1: figure (or, for Y categorical x X categorical,
#     two tables side by side in the same position).
#   Row 1, column 2: regression table, Y as a function of X.
#   Row 2, column 1: SD(Y), SD(X), correlation, covariance, and
#     residual variance RSS / (n - p), where p is the number of
#     estimated coefficients including the intercept.
#   Row 2, column 2: rho = cov(Y,X) / var(X) and
#     beta = corr(Y,X) * (SD(Y) / SD(X)), with Y replaced by
#     logit(Y) at group level when Y is dichotomous (see the
#     rho/beta panel code).
#
# Decisions and definitions, stated here so the app is not ambiguous:
# * Variables: pov_inc_ratio, bmi and phq9 are continuous. All other
#   variables in the file are categorical. gender (1/2), male (0/1),
#   less_9, nine_11, hs_ged, scoll, coll and depression (0/1) are
#   binary; education (5 levels) and ethnicity (6 levels) are
#   multi-category. Binary Y is the only case in which a logit /
#   logistic regression is defined; see below.
# * Complete cases: every panel uses the observations with both the
#   chosen Y and X observed, exactly as Stata's regress / logit /
#   summarize do. N and the number excluded are shown above the panels.
#   (The file name "noNAs" is misleading: pov_inc_ratio is missing
#   for 364 rows, phq9 for 443, education for 3 and bmi for 48.)
# * "Standard deviation of Y, standard deviation of Y" in the task is
#   read as SD(Y) and SD(X); the second one is X, as in the do file's
#   sd_phq9 and sd_PIR.
# * Row 2, column 2: beta = corr * (SD(Y) / SD(X)), which equals
#   rho = cov / var(X) and the regression slope for a continuous X.
# * Residual variance in row 2, column 1 is RSS / (n - p), where
#   p is the number of estimated coefficients including the
#   intercept -- the usual residual-variance estimator, whose square
#   root is lm's Root MSE shown in the regression panel. (An earlier
#   version used RSS / (N - 1); the user corrected that to n - p.)
#   For a binary Y, the regression panel is logistic (no RSS);
#   the RSS / (n - p) shown in row 2, column 1 is then from the
#   linear-probability fit and is labelled as such.
# * Regression table terms: the internal names "x_raw" and
#   "factor(x_raw)k" are replaced by the variable name for a
#   continuous X and by "variable: category label" for a factor X
#   (e.g. "education: HS/GED"); "(Intercept)" is kept as it is.
# * Y categorical x X categorical, right-hand table: if Y is not
#   binary, for each cell (i, j) the odds ratio is the 2x2 odds
#   ratio obtained by collapsing the rest of the table:
#   (n_ij * n_not-i,not-j) / (n_i,not-j * n_not-i,j). In a 2x2 table
#   this gives the odds ratio for the diagonal cells and its
#   reciprocal for the off-diagonal cells, because flipping either
#   the row or the column inverts an odds ratio.
#   If Y is binary, the table instead uses the reference category
#   chosen above the regression table: for cell (i, j), OR = odds
#   of Y = row i in X = column j, divided by odds of Y = row i in
#   the reference column. The reference column is 1, and the row
#   for the Y category coded 1 matches the logistic regression
#   odds ratios.
# * Y categorical x X continuous, logit plot: logit is only defined
#   for a binary Y. X is split into (up to) 10 quantile bins; for each
#   bin the app plots the empirical logit log((y + 0.5) / (n - y + 0.5))
#   against the bin mean of X, with point size proportional to n,
#   plus the fitted log-odds line from glm(Y ~ X). Plotting logit of
#   an individual 0/1 value would be -Inf / +Inf, so the binned
#   empirical logit is the plot that can actually be drawn.
#   For a multi-category Y (education, ethnicity) with a continuous X,
#   the app says so and does not invent a logit.
# * Y continuous x X categorical, whisker-only plot (no bars):
#   point = mean of Y in the X category, whisker = mean +/- 1 SE,
#   label = N in the category, y-axis zoomed to the whisker range
#   rather than starting at zero.
# * Regression panel: continuous Y -> lm; binary Y -> glm binomial.
#   For a categorical X in a linear model it uses factor(X),
#   base = first level, so coefficients are differences from the
#   base category, as in Stata's i. coding. For a categorical X in
#   a logistic model the app asks for a reference category (a
#   selector above the regression table, default = first level),
#   and the odds ratios are relative to that chosen category.
#   For a multi-category Y the panel says that binary logistic
#   regression does not apply.

library(shiny)
library(data.table)
library(ggplot2)

# ---------------------------------------------------------------- data
dt_all <- fread("data_for_class.csv")

# Variables in the file, in file order
all_vars <- c("gender", "pov_inc_ratio", "education", "bmi", "phq9",
              "ethnicity", "male", "less_9", "nine_11", "hs_ged",
              "scoll", "coll", "depression")
# Variables available for selection (user restriction):
# Y (outcome) only: phq9, bmi, depression, pov_inc_ratio.
# X (predictor): all variables except the education dummies
#   less_9, nine_11, hs_ged, scoll, coll.
y_vars <- c("phq9", "bmi", "depression", "pov_inc_ratio")
x_vars <- setdiff(all_vars, c("less_9", "nine_11", "hs_ged",
                              "scoll", "coll"))
continuous_vars <- c("pov_inc_ratio", "bmi", "phq9")
categorical_vars <- setdiff(all_vars, continuous_vars)
binary_vars <- c("gender", "male", "less_9", "nine_11", "hs_ged",
                 "scoll", "coll", "depression")

# Human-readable labels for categorical levels (numeric code -> label)
var_labels <- list(
  gender = c("1" = "Male", "2" = "Female"),
  education = c("1" = "< 9th grade", "2" = "9th - 11th", "3" = "HS/GED",
                "4" = "Some College/AA", "5" = "College Grad+"),
  ethnicity = c("1" = "Mexican American", "2" = "Other Hispanic",
                "3" = "NH White", "4" = "NH Black", "5" = "NH Asian",
                "6" = "Other/Multi"),
  male = c("0" = "Female", "1" = "Male"),
  less_9 = c("0" = "No", "1" = "Yes (< 9th)"),
  nine_11 = c("0" = "No", "1" = "Yes (9th-11th)"),
  hs_ged = c("0" = "No", "1" = "Yes (HS/GED)"),
  scoll = c("0" = "No", "1" = "Yes (Some College)"),
  coll = c("0" = "No", "1" = "Yes (College Grad+)"),
  depression = c("0" = "Not depressed (0-9)", "1" = "Depressed (10-27)")
)

label_for <- function(var, value) {
  labs <- var_labels[[var]]
  if (is.null(labs)) return(as.character(value))
  out <- labs[as.character(value)]
  out[is.na(out)] <- as.character(value[is.na(out)])
  unname(out)
}

# Binary 0/1 coding used for logit / logistic / point-biserial work.
# gender: 1 = Female (2), 0 = Male (1), matching the do file's female
# dummy and its logit model (base = Male). All other binary variables
# are already 0/1 in the file.
binary_code <- function(var, x) {
  if (identical(var, "gender")) {
    as.numeric(x == 2)
  } else {
    as.numeric(x)
  }
}

# ---------------------------------------------------------------- UI
ui <- fluidPage(
  titlePanel("Bivariate analysis: Y vs X"),
  p("Data: NHANES 2013-2014 extract (N = 4,721 rows). Every panel uses complete cases ",
    "for the chosen Y and X. Y-variable (outcome): phq9, bmi, ",
    "depression or pov_inc_ratio only. X-variable (predictor): any ",
    "variable except less_9, nine_11, hs_ged, scoll and coll. ",
    "Continuous variables: pov_inc_ratio, bmi, phq9. The others are ",
    "categorical; gender, male and depression are binary."),
  fluidRow(
    column(6, selectInput("yvar", "Y-variable (outcome)",
                          choices = y_vars, selected = "phq9")),
    column(6, selectInput("xvar", "X-variable (predictor)",
                          choices = x_vars, selected = "pov_inc_ratio"))
  ),
  textOutput("sample_info"),
  hr(),
  # Row 1
  fluidRow(
    column(6,
           h4("Figure: Y vs X"),
           uiOutput("figure_panel")),
    column(6,
           h4("Regression: Y as a function of X"),
           uiOutput("ref_ui"),
           verbatimTextOutput("reg_header"),
           tableOutput("reg_table"))
  ),
  hr(),
  # Row 2
  fluidRow(
    column(6,
           h4("Summary statistics (complete cases)"),
           tableOutput("summary_table"),
           p("Residual variance is RSS / (n - p), where p is the number ",
             "of estimated coefficients including the intercept. ",
             "For a binary Y the regression above is logistic, which ",
             "has no RSS; the RSS shown here is then from the ",
             "linear-probability fit and is labelled as such.")),
    column(6,
           h4("Slope by hand: a and beta: y = alpha + a*X + e_i; or y = alpha + beta*X + e_i"),
           uiOutput("rho_beta_panel"))
  )
)

# ---------------------------------------------------------------- server
server <- function(input, output, session) {

  # Complete-case data.table for the chosen pair, with labelled
  # categorical values and numeric codes kept alongside.
  dat <- reactive({
    req(input$yvar, input$xvar)
    validate(need(input$yvar != input$xvar,
                  "Please choose different variables for Y and X."))
    yv <- input$yvar
    xv <- input$xvar
    d <- dt_all[!is.na(get(yv)) & !is.na(get(xv)),
                .(y_raw = get(yv), x_raw = get(xv))]
    d[, y_type := if (yv %in% continuous_vars) "continuous" else "categorical"]
    d[, x_type := if (xv %in% continuous_vars) "continuous" else "categorical"]
    d[, y_lab := if (yv %in% categorical_vars) label_for(yv, y_raw) else as.character(y_raw)]
    d[, x_lab := if (xv %in% categorical_vars) label_for(xv, x_raw) else as.character(x_raw)]
    # Keep categorical levels in numeric-code order (not alphabetical
    # order) for the counts / odds-ratio tables and the whisker plot:
    # e.g. education <9th, 9-11, HS/GED, Some College, College Grad+,
    # not the alphabetical order dcast would otherwise give.
    if (yv %in% categorical_vars) {
      d[, y_lab := factor(y_lab, levels = label_for(yv, sort(unique(y_raw))))]
    }
    if (xv %in% categorical_vars) {
      d[, x_lab := factor(x_lab, levels = label_for(xv, sort(unique(x_raw))))]
    }
    d[, y_bin := binary_code(yv, y_raw)]
    d[, x_bin := binary_code(xv, x_raw)]
    # Numeric versions for correlation / slope work: continuous values
    # as stored, binary variables as 0/1, multi-category as NA (not a
    # meaningful numeric scale).
    d[, y_num := if (yv %in% continuous_vars) y_raw else
      if (yv %in% binary_vars) y_bin else NA_real_]
    d[, x_num := if (xv %in% continuous_vars) x_raw else
      if (xv %in% binary_vars) x_bin else NA_real_]
    d
  })

  y_type <- reactive({
    if (input$yvar %in% continuous_vars) "continuous" else "categorical"
  })
  x_type <- reactive({
    if (input$xvar %in% continuous_vars) "continuous" else "categorical"
  })
  y_is_binary <- reactive(input$yvar %in% binary_vars)
  x_is_binary <- reactive(input$xvar %in% binary_vars)

  fig_type <- reactive({
    yt <- y_type()
    xt <- x_type()
    if (yt == "continuous" && xt == "continuous") "scatter"
    else if (yt == "continuous" && xt == "categorical") "whisker"
    else if (yt == "categorical" && xt == "categorical") "catcat"
    else if (yt == "categorical" && xt == "continuous") {
      if (y_is_binary()) "logit" else "logit_multi"
    } else "unknown"
  })

  output$sample_info <- renderText({
    req(input$yvar, input$xvar)
    n_all <- nrow(dt_all)
    n_cc <- nrow(dat())
    sprintf("Y = %s (%s), X = %s (%s). Complete cases: N = %d of %d rows; %d excluded for missing Y or X.",
            input$yvar, y_type(), input$xvar, x_type(),
            n_cc, n_all, n_all - n_cc)
  })

  # ---- Row 1, column 1: figure, or two tables for cat x cat ----
  output$figure_panel <- renderUI({
    ft <- fig_type()
    if (ft == "catcat") {
      fluidRow(
        column(6, h5("Counts: Y rows x X columns"), tableOutput("counts_table")),
        column(6, h5("Odds ratio per cell"), tableOutput("or_table"),
               if (y_is_binary()) {
                 p(paste0("ORs are relative to the reference category for ",
                          input$xvar, ": ", label_for(input$xvar, x_ref_code()),
                          " (chosen above the regression table). For each cell, ",
                          "OR = odds of Y = row in X = column, divided by odds ",
                          "of Y = row in the reference category. The reference ",
                          "column is therefore 1.000 in both rows; the row for ",
                          "the Y category coded 1 matches the logistic ",
                          "regression odds ratios, and the other row gives ",
                          "their reciprocals."))
               } else {
                 p("Cell OR: odds of Y = row vs. not row, for X = column ",
                   "vs. not column, collapsing the rest of the table to ",
                   "2x2. In a 2x2 table the diagonal cells give the OR ",
                   "and the off-diagonal cells give its reciprocal.")
               })
      )
    } else if (ft == "logit_multi") {
      p(sprintf(paste0(
        "Logit is only defined for a binary Y. %s has %d categories, ",
        "so there is no single logit(Y) to plot against %s. ",
        "Choose depression as Y for the logit plot, or use a ",
        "categorical X to see the counts and odds-ratio tables."),
        input$yvar,
        length(unique(dat()$y_raw)), input$xvar))
    } else {
      plotOutput("figure_plot", height = "380px")
    }
  })

  output$figure_plot <- renderPlot({
    d <- dat()
    ft <- fig_type()
    yv <- input$yvar
    xv <- input$xvar
    if (ft == "scatter") {
      ggplot(d, aes(x = x_raw, y = y_raw)) +
        geom_point(alpha = 0.35) +
        geom_smooth(method = "lm", se = FALSE, colour = "red") +
        labs(x = xv, y = yv, title = paste(yv, "vs", xv),
             subtitle = "Red line: OLS fit")
    } else if (ft == "whisker") {
      # Whisker-only plot (no bars, by user request): a point at the
      # mean of Y in each X category, with a vertical whisker
      # mean +/- 1 SE. Removing the bars also removes ggplot's
      # bar-driven y = 0 baseline, so the y-axis is zoomed to the
      # whisker range instead of starting at zero.
      summ <- d[, .(mean_y = mean(y_raw), se_y = sd(y_raw) / sqrt(.N),
                    n = .N), by = x_lab]
      # Keep X categories in numeric-code order, not alphabetical order
      summ[, x_ord := match(x_lab, unique(d[order(x_raw)]$x_lab))]
      setorder(summ, x_ord)
      lo <- min(summ$mean_y - summ$se_y)
      hi <- max(summ$mean_y + summ$se_y)
      pad <- if (hi > lo) 0.15 * (hi - lo) else 0.5
      ggplot(summ, aes(x = reorder(x_lab, x_ord), y = mean_y)) +
        geom_errorbar(aes(ymin = mean_y - se_y, ymax = mean_y + se_y),
                      width = 0.2) +
        geom_point(size = 3) +
        geom_text(aes(y = mean_y + se_y, label = paste0("N = ", n)),
                  vjust = -0.5, size = 3.5) +
        coord_cartesian(ylim = c(lo - pad, hi + pad)) +
        labs(x = xv, y = paste("Mean", yv),
             title = paste("Mean", yv, "by", xv),
             subtitle = "Points: mean. Whiskers: mean +/- 1 SE. Y-axis zoomed to the whisker range.")
    } else if (ft == "logit") {
      # Empirical logit by quantile bins of X
      d2 <- d[order(x_raw)]
      # Up to 10 quantile bins; fall back to unique X values if needed
      qs <- unique(quantile(d2$x_raw, probs = seq(0, 1, by = 0.1),
                            na.rm = TRUE, names = FALSE))
      if (length(qs) < 3L) {
        d2[, bin := x_raw]
      } else {
        d2[, bin := cut(x_raw, breaks = qs, include.lowest = TRUE)]
      }
      bins <- d2[, .(mean_x = mean(x_raw), n = .N,
                     y_sum = sum(y_bin)), by = bin]
      bins[, p := y_sum / n]
      bins[, logit_emp := log((y_sum + 0.5) / (n - y_sum + 0.5))]
      # Fitted log-odds line from glm(Y ~ X)
      fit <- glm(y_bin ~ x_raw, data = d, family = binomial())
      grid <- data.table(x_raw = seq(min(d$x_raw), max(d$x_raw),
                                     length.out = 100))
      grid[, logit_fit := predict(fit, newdata = grid, type = "link")]
      ggplot() +
        geom_point(data = bins, aes(x = mean_x, y = logit_emp, size = n),
                   alpha = 0.7) +
        geom_line(data = grid, aes(x = x_raw, y = logit_fit),
                  colour = "red") +
        labs(x = xv, y = paste0("logit(", yv, ")"),
             title = paste("Empirical logit of", yv, "vs", xv),
             subtitle = paste0("Points: empirical logit in X bins (size = n). ",
                               "Red line: fitted log-odds from glm(", yv, " ~ ", xv, ")."))
    } else {
      ggplot() + labs(title = "No figure for this combination")
    }
  })

  output$counts_table <- renderTable({
    d <- dat()
    ct <- dcast(d, y_lab ~ x_lab, fun.aggregate = length, value.var = "y_raw")
    setnames(ct, "y_lab", paste0(input$yvar, " \\ ", input$xvar))
    ct
  }, rownames = FALSE, digits = 0)

  output$or_table <- renderTable({
    d <- dat()
    ct <- d[, .N, by = .(y_lab, x_lab)]
    mat <- dcast(ct, y_lab ~ x_lab, value.var = "N", fill = 0)
    y_levels <- mat$y_lab
    x_levels <- setdiff(names(mat), "y_lab")
    m <- as.matrix(mat[, ..x_levels])
    rownames(m) <- y_levels
    or_mat <- matrix(NA_real_, nrow = nrow(m), ncol = ncol(m),
                     dimnames = list(y_levels, x_levels))
    total <- sum(m)
    for (i in seq_len(nrow(m))) {
      for (j in seq_len(ncol(m))) {
        a <- m[i, j]
        b <- sum(m[i, ]) - a
        c_ <- sum(m[, j]) - a
        dd <- total - a - b - c_
        or_mat[i, j] <- if (b * c_ == 0) {
          if (a * dd == 0) NA_real_ else Inf
        } else {
          (a * dd) / (b * c_)
        }
      }
    }
    # If Y is binary, replace the cell-vs-rest ORs above with
    # reference-category ORs, using the reference chosen above the
    # regression table (x_ref_code is defined in the regression
    # section below, but is available here at run time, since the
    # whole server function has run before any reactive fires).
    # For cell (i, j): OR = odds of Y = row i in X = column j,
    # divided by odds of Y = row i in the reference column.
    # The reference column is therefore 1 in both rows; the row
    # for the Y category coded 1 in the logistic model matches
    # the regression odds ratios exactly, and the other row gives
    # their reciprocals. If Y is not binary, the cell-vs-rest ORs
    # above are kept, since there is no single reference OR.
    ref <- x_ref_code()
    if (nrow(m) == 2L && !is.null(ref)) {
      ref_lab <- label_for(input$xvar, ref)
      j_ref <- match(ref_lab, x_levels)
      if (!is.na(j_ref)) {
        odds_of <- function(num, den) {
          if (den == 0) { if (num == 0) NA_real_ else Inf } else num / den
        }
        for (i in seq_len(nrow(m))) {
          i_other <- setdiff(seq_len(nrow(m)), i)
          odds_ref <- odds_of(m[i, j_ref], m[i_other, j_ref])
          for (j in seq_len(ncol(m))) {
            odds_j <- odds_of(m[i, j], m[i_other, j])
            or_mat[i, j] <- if (is.na(odds_ref) || is.na(odds_j)) {
              NA_real_
            } else if (odds_ref == 0) {
              if (odds_j == 0) NA_real_ else Inf
            } else if (is.infinite(odds_ref)) {
              if (is.infinite(odds_j)) NA_real_ else 0
            } else {
              odds_j / odds_ref
            }
          }
        }
      }
    }
    out <- data.table(y = y_levels, as.data.table(round(or_mat, 3)))
    setnames(out, "y", paste0(input$yvar, " \\ ", input$xvar))
    out
  }, rownames = FALSE, digits = 3)

  # ---- Reference category for the logistic regression ----
  # Only when Y is dichotomous and X is categorical: the odds
  # ratios in the regression panel AND in the contingency-table
  # odds-ratio panel are relative to this category.
  # The default is the first (lowest-code) category, which is the
  # previous behaviour. A stale or missing input value falls back
  # to that first category, so switching X cannot leave the fit
  # with an empty reference level.
  x_ref_code <- reactive({
    if (!y_is_binary() || x_type() != "categorical") return(NULL)
    d <- dat()
    lvls <- sort(unique(d$x_raw))
    if (length(lvls) == 0) return(NULL)
    ref_raw <- input$ref_cat
    ref <- if (is.null(ref_raw)) lvls[1] else suppressWarnings(as.numeric(ref_raw))
    if (length(ref) != 1 || is.na(ref) || !(ref %in% lvls)) lvls[1] else ref
  })

  output$ref_ui <- renderUI({
    if (!y_is_binary() || x_type() != "categorical") return(NULL)
    d <- dat()
    lvls <- sort(unique(d$x_raw))
    labs <- label_for(input$xvar, lvls)
    selectInput("ref_cat",
                "Reference category for X (odds ratios are relative to this category):",
                choices = setNames(as.character(lvls), labs),
                selected = as.character(lvls[1]))
  })

  # ---- Row 1, column 2: regression table ----
  reg_fit <- reactive({
    d <- dat()
    yv <- input$yvar
    xv <- input$xvar
    if (y_type() == "continuous") {
      if (x_type() == "continuous") {
        lm(y_raw ~ x_raw, data = d)
      } else {
        lm(y_raw ~ factor(x_raw), data = d)
      }
    } else if (y_is_binary()) {
      if (x_type() == "continuous") {
        glm(y_bin ~ x_raw, data = d, family = binomial())
      } else {
        # Factor with the chosen reference first, so all other
        # categories are compared with it. Using a named x_factor
        # column (instead of factor() in the formula) keeps the
        # coefficient names simple ("x_factor<level>") for the
        # term-renaming in reg_table below.
        ref <- x_ref_code()
        lvls <- sort(unique(d$x_raw))
        levels_order <- c(ref, setdiff(lvls, ref))
        d2 <- copy(d)
        d2[, x_factor := factor(x_raw, levels = levels_order)]
        glm(y_bin ~ x_factor, data = d2, family = binomial())
      }
    } else {
      NULL
    }
  })

  output$reg_header <- renderText({
    fit <- reg_fit()
    d <- dat()
    if (is.null(fit)) {
      sprintf(paste0("Logistic regression is binary. %s has %d categories, ",
                     "so no single logistic model is fitted here. ",
                     "Choose a binary Y for a logistic model."),
              input$yvar, length(unique(d$y_raw)))
    } else if (inherits(fit, "lm") && !inherits(fit, "glm")) {
      # NOTE: a glm object also inherits from lm in R, so the glm
      # check must come first / be excluded here, otherwise a
      # logistic fit would fall into the linear branch and fail
      # on summary(fit)$fstatistic, which is NULL for a glm.
      s <- summary(fit)
      f <- s$fstatistic
      sprintf(paste0("Linear regression: %s ~ %s\nN = %d, R-squared = %.4f, ",
                     "adj. R-squared = %.4f, Root MSE = %.4f, ",
                     "F(%d, %d) = %.2f, Prob > F = %.4f"),
              input$yvar, input$xvar, nobs(fit), s$r.squared,
              s$adj.r.squared, s$sigma, f[2], f[3], f[1],
              pf(f[1], f[2], f[3], lower.tail = FALSE))
    } else {
      ll <- as.numeric(logLik(fit))
      null_fit <- glm(y_bin ~ 1, data = d, family = binomial())
      ll0 <- as.numeric(logLik(null_fit))
      lr <- 2 * (ll - ll0)
      df_lr <- length(coef(fit)) - 1L
      base_txt <- sprintf(paste0("Logistic regression: %s ~ %s (log-odds coefficients, ",
                                 "with odds ratios)\nN = %d, Log likelihood = %.4f, ",
                                 "Pseudo R2 = %.4f, LR chi2(%d) = %.2f, Prob > chi2 = %.4f"),
                          input$yvar, input$xvar, nobs(fit), ll, 1 - ll / ll0,
                          df_lr, lr, pchisq(lr, df_lr, lower.tail = FALSE))
      if (x_type() == "categorical") {
        ref_lab <- label_for(input$xvar, x_ref_code())
        paste0(base_txt, "\nReference category for ", input$xvar, ": ",
               ref_lab, " (odds ratios are relative to this category).")
      } else {
        base_txt
      }
    }
  })

  output$reg_table <- renderTable({
    fit <- reg_fit()
    if (is.null(fit)) return(data.table(Message = "No binary logistic model for a multi-category Y."))
    s <- summary(fit)
    co <- as.data.table(s$coefficients, keep.rownames = "term")
    # glm also inherits from lm, so exclude glm from the lm branch
    if (inherits(fit, "lm") && !inherits(fit, "glm")) {
      setnames(co, c("term", "estimate", "std_error", "statistic", "p_value"))
      ci <- as.data.table(confint(fit), keep.rownames = "term")
      setnames(ci, c("term", "ci_low", "ci_high"))
      co <- merge(co, ci, by = "term", sort = FALSE)
      setnames(co, "statistic", "t")
    } else {
      setnames(co, c("term", "estimate", "std_error", "statistic", "p_value"))
      ci <- as.data.table(confint.default(fit), keep.rownames = "term")
      setnames(ci, c("term", "ci_low", "ci_high"))
      co <- merge(co, ci, by = "term", sort = FALSE)
      setnames(co, "statistic", "z")
      co[, odds_ratio := exp(estimate)]
      co[, or_low := exp(ci_low)]
      co[, or_high := exp(ci_high)]
    }
    # Rename the internal model terms to readable labels, after the
    # merge above (which needs the original term names to match).
    # "(Intercept)" stays as it is; a continuous X becomes its
    # variable name (e.g. "pov_inc_ratio" instead of "x_raw"); a
    # factor X becomes "variable: category label" (e.g.
    # "education: HS/GED" instead of "factor(x_raw)3", "gender:
    # Female" instead of "factor(x_raw)2"). For the logistic fit
    # the factor column is x_factor, so its terms are "x_factor<level>".
    # The base / reference level is the first (lowest-code) level
    # for linear fits, and the chosen reference category for
    # logistic fits (see x_ref_code above).
    xv <- input$xvar
    pretty_term <- function(term) {
      if (term == "(Intercept)") return(term)
      if (term == "x_raw") return(xv)
      if (grepl("^factor\\(x_raw\\)", term)) {
        lvl <- sub("^factor\\(x_raw\\)", "", term)
        return(paste0(xv, ": ", label_for(xv, as.numeric(lvl))))
      }
      if (grepl("^x_factor", term)) {
        lvl <- sub("^x_factor", "", term)
        return(paste0(xv, ": ", label_for(xv, as.numeric(lvl))))
      }
      term
    }
    co[, term := vapply(term, pretty_term, character(1))]
    # Round for display; the underlying fit is unrounded
    num_cols <- setdiff(names(co), "term")
    co[, (num_cols) := lapply(.SD, function(x) round(x, 3)), .SDcols = num_cols]
    co
  }, rownames = FALSE, digits = 3)

  # ---- Row 2, column 1: summaries ----
  output$summary_table <- renderTable({
    d <- dat()
    n <- nrow(d)
    # SDs: continuous as stored; binary as 0/1 coding; multi-category: NA
    sd_y <- if (y_type() == "continuous") sd(d$y_raw) else
      if (y_is_binary()) sd(d$y_bin) else NA_real_
    sd_x <- if (x_type() == "continuous") sd(d$x_raw) else
      if (x_is_binary()) sd(d$x_bin) else NA_real_
    # Correlation / covariance only on a meaningful numeric scale
    cor_yx <- if (!is.na(sd_y) && !is.na(sd_x)) cor(d$y_num, d$x_num) else NA_real_
    cov_yx <- if (!is.na(sd_y) && !is.na(sd_x)) cov(d$y_num, d$x_num) else NA_real_
    # Residual variance RSS / (n - p), p = number of coefficients
    # including the intercept (corrected from RSS / (N - 1)).
    rss <- NA_real_
    rss_label <- "RSS / (n - p)"
    if (y_type() == "continuous") {
      fit <- reg_fit()
      rss <- sum(residuals(fit)^2) / (n - length(coef(fit)))
      rss_label <- sprintf("RSS / (n - p) [p = %d]", length(coef(fit)))
    } else if (y_is_binary()) {
      # Linear-probability fit, for reference only; the logistic fit
      # in row 1, column 2 has no RSS.
      fit_lp <- if (x_type() == "continuous") {
        lm(y_bin ~ x_raw, data = d)
      } else {
        lm(y_bin ~ factor(x_raw), data = d)
      }
      rss <- sum(residuals(fit_lp)^2) / (n - length(coef(fit_lp)))
      rss_label <- sprintf("RSS / (n - p) [p = %d] (linear-probability fit, reference only)",
                           length(coef(fit_lp)))
    }
    fmt <- function(x) if (is.na(x)) "—" else sprintf("%.3f", x)
    data.table(
      Quantity = c("N (complete cases)", "SD(Y)", "SD(X)",
                   "Correlation(Y, X)", "Covariance(Y, X)", rss_label),
      Value = c(as.character(n), fmt(sd_y), fmt(sd_x),
                fmt(cor_yx), fmt(cov_yx), fmt(rss))
    )
  }, rownames = FALSE, digits = 3)

  # ---- Row 2, column 2: rho and beta ----
  # For a continuous Y, rho/beta are computed from Y as stored.
  # For a dichotomous Y, they are computed from logit(Y), not
  # from the 0/1 Y, because the logistic model is linear in the
  # logits. A single logit value per observation does not exist
  # (logit(0) = -Inf, logit(1) = +Inf), so the logit is taken at
  # group level, exactly as in the logit figure:
  #  - X categorical (binary): one group per X category, logit =
  #    log(p / (1 - p)) with p = proportion Y = 1 in the category
  #    (exact, so rho = beta = the log odds ratio, matching the
  #    logistic coefficient). X is the binary coding (e.g. gender:
  #    Male = 0, Female = 1).
  #  - X continuous: X quantile bins (up to 10, the same bins as
  #    the logit figure), logit = empirical logit
  #    log((y_sum + 0.5) / (n - y_sum + 0.5)) in each bin, X =
  #    bin mean X. This binned slope approximates, but is not
  #    identical to, the logistic slope in the regression panel.
  # The SD / covariance / correlation below are therefore for
  # logit(Y) at group level when Y is binary, not for the 0/1 Y
  # shown in the summary panel (row 2, column 1). For a
  # multi-category X there is no single slope, so no rho/beta
  # is shown; the per-category logit slopes are in the
  # regression table, relative to the reference category.
  output$rho_beta_panel <- renderUI({
    d <- dat()
    n <- nrow(d)
    if (is.na(d$y_num[1]) || is.na(d$x_num[1]) ||
        all(is.na(d$y_num)) || all(is.na(d$x_num))) {
      return(p(paste0("rho and beta are single slopes, so they are only ",
                      "defined when X is numeric (continuous or binary) ",
                      "and Y is numeric (continuous or binary). Here X = ",
                      input$xvar, " and Y = ", input$yvar,
                      " do not both meet that condition, so no single ",
                      "rho / beta is shown. ",
                      if (y_is_binary()) {
                        "For a binary Y, the per-category logit slopes, relative to the reference category, are in the regression table above."
                      } else {
                        "See the regression table above for the factor-X coefficients."
                      })))
    }
    if (y_is_binary()) {
      y_lab <- "logit(Y)"
      if (x_type() == "categorical") {
        # One group per X category, exact category logit
        g <- d[, .(n_grp = .N, y_sum = sum(y_bin)), by = .(x_val = x_bin, x_lab)]
        setorder(g, x_val)
        g[, p_grp := y_sum / n_grp]
        # Exact logit, except for a 0 or 1 proportion, where the
        # +0.5 empirical logit is used so the value stays finite
        g[, logit_grp := ifelse(p_grp > 0 & p_grp < 1,
                                log(p_grp / (1 - p_grp)),
                                log((y_sum + 0.5) / (n_grp - y_sum + 0.5)))]
        g_x <- g$x_val
        g_y <- g$logit_grp
        logit_note <- paste0("logit(", input$yvar, ") by ", input$xvar, ": ",
                             paste0(g$x_lab, ": ", sprintf("%.3f", g$logit_grp),
                                    collapse = ", "), ".")
        extra_note <- paste0("Correlation is 1.0000 in absolute value ",
                             "because logit(Y) is constant within each X ",
                             "category, so it is exactly linear in the ",
                             "binary X.")
        n_note <- paste0("Groups: ", nrow(g), " X categories, from N = ",
                         n, " complete cases.")
        glm_note <- NULL
      } else {
        # X continuous: quantile bins, same as the logit figure
        d2 <- d[order(x_raw)]
        qs <- unique(quantile(d2$x_raw, probs = seq(0, 1, by = 0.1),
                              na.rm = TRUE, names = FALSE))
        if (length(qs) < 3L) {
          d2[, bin := x_raw]
        } else {
          d2[, bin := cut(x_raw, breaks = qs, include.lowest = TRUE)]
        }
        bins <- d2[, .(mean_x = mean(x_raw), n_grp = .N,
                       y_sum = sum(y_bin)), by = bin]
        bins[, logit_grp := log((y_sum + 0.5) / (n_grp - y_sum + 0.5))]
        g_x <- bins$mean_x
        g_y <- bins$logit_grp
        logit_note <- paste0("logit(", input$yvar, ") is the empirical logit ",
                             "in ", nrow(bins), " X quantile bins (the same ",
                             "bins as the logit figure).")
        extra_note <- NULL
        n_note <- paste0("Groups: ", nrow(bins), " X bins, from N = ",
                         n, " complete cases.")
        fit <- reg_fit()
        glm_note <- if (!is.null(fit) && length(coef(fit)) >= 2L) {
          paste0("For comparison, the logistic slope in the regression ",
                 "panel is ", sprintf("%.3f", coef(fit)[2]),
                 ". The binned-logit slope above approximates it; it is ",
                 "not identical because the logit is binned and uses ",
                 "the +0.5 correction.")
        } else {
          NULL
        }
      }
      sd_y <- sd(g_y)
      sd_x <- sd(g_x)
      cov_yx <- cov(g_y, g_x)
      cor_yx <- cor(g_y, g_x)
      var_x <- var(g_x)
    } else {
      y_lab <- "Y"
      sd_y <- sd(d$y_num)
      sd_x <- sd(d$x_num)
      cov_yx <- cov(d$y_num, d$x_num)
      cor_yx <- cor(d$y_num, d$x_num)
      var_x <- var(d$x_num)
      logit_note <- NULL
      extra_note <- NULL
      n_note <- paste0("N (complete cases) = ", n, ".")
      glm_note <- NULL
    }
    rho <- cov_yx / var_x
    beta <- cor_yx * (sd_y / sd_x)
    # Avoid a "-0.000000" display when the difference is a tiny
    # negative floating-point residue: at 6 decimals it is 0.
    diff_val <- beta - rho
    if (abs(diff_val) < 5e-7) diff_val <- 0
    tagList(
      if (!is.null(logit_note)) p(logit_note),
      p(sprintf("a = cov(%s, X) / var(X) = %.3f / %.3f = %.3f",
                y_lab, cov_yx, var_x, rho)),
      p(sprintf("beta = corr(%s, X) x (SD(%s) / SD(X)) = %.3f x (%.3f / %.3f) = %.3f",
                y_lab, y_lab, cor_yx, sd_y, sd_x, beta)),
      # NOTE: the "because ..." text is part of the format string,
      # not extra sprintf arguments. Passing it as an argument made
      # sprintf try to format a character string with %.6f, giving
      # "invalid format '%.6f'; use format %s for character objects".
      p(sprintf(paste0("Difference (beta - a) = %.6f. They agree ",
                       "(up to rounding) because corr x SD(%s)/SD(X) is ",
                       "algebraically the same as cov / var(X)."),
                diff_val, y_lab)),
      if (!is.null(extra_note)) p(extra_note),
      if (!is.null(glm_note)) p(glm_note),
      p(n_note)
    )
  })
}

shinyApp(ui = ui, server = server)
