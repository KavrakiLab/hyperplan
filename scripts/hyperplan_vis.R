#!/usr/bin/env Rscript
######################################################################
# Software License Agreement (BSD License)
#
#  Copyright (c) 2020, Rice University
#  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions
#  are met:
#
#   * Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#   * Redistributions in binary form must reproduce the above
#     copyright notice, this list of conditions and the following
#     disclaimer in the documentation and/or other materials provided
#     with the distribution.
#   * Neither the name of Rice University nor the names of its
#     contributors may be used to endorse or promote products derived
#     from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
#  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
#  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
#  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
#  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
#  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
#  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.
######################################################################

library(dplyr, warn.conflicts = FALSE, quietly = TRUE)
library(ggplot2)
library(ggthemes)
library(ggbeeswarm)

cols <- c(few_pal("Medium")(8), rep("#808080", 10))

read_run <- function(dir) {
    df <- read.csv(paste(dir, "/results.csv", sep = ""), na.strings = c("NA", "None", "NaN", "nan"))
    df$run.id <- basename(dir)
    df
}

generate_plots <- function(dirs) {
    dfs <- lapply(dirs, read_run)
    df <- bind_rows(dfs)
    df$budget <- factor(signif(df$budget, 3))
    df$model_based <- factor(df$model_based, labels = c("random", "model-based"))
    df <- df %>%
        add_count(planner) %>%
        transform(planner = reorder(planner, -n))
    loss <- ggplot(df, aes(x = budget, y = loss)) +
        geom_quasirandom(aes(color = planner, alpha = I(.5))) +
        scale_y_log10(
            breaks = c(1, 10, 100, 1000, 10000, 100000),
            labels = c("1", "10", "100", "1000", "10000", "100000")
        ) +
        xlab("budget (seconds)") +
        theme_tufte() +
        scale_color_manual(values = cols)
    if (length(dirs) > 1) {
        loss <- loss + facet_grid(rows = vars(run.id))
    }
    ggsave("loss.pdf", width = 6.5, height = 3, units = "in", scale = 1.3)
    #    planners <- ggplot(df, aes(budget)) +
    #        geom_bar(aes(fill=planner), position="fill") +
    #        xlab('budget (s)') +
    #        ylab('fraction') +
    #        theme_tufte() +
    #        scale_fill_manual(values=cols)
    #    ggsave(paste(dir, '/planners.png', sep=""))
    #    model_based <- ggplot(df, aes(budget)) +
    #        geom_bar(aes(fill=model_based), position="fill") +
    #        xlab('budget (s)') +
    #        ylab('fraction') +
    #        labs(fill="") +
    #        theme_tufte() +
    #        scale_fill_manual(values=cols)
    #    ggsave(paste(dir, '/model_based.png', sep=""))
}

generate_plots(commandArgs(trailingOnly = TRUE))