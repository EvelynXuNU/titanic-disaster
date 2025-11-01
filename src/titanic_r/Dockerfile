# Use R base image
FROM r-base:4.3.0

# Copy project files
COPY . .

# Install R packages
RUN Rscript -e "install.packages(c('tidyverse', 'caret'), repos='https://cloud.r-project.org', dependencies=TRUE, clean=TRUE)"

# Run the R script
CMD ["Rscript", "main.R"]