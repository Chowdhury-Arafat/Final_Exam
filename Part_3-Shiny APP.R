library(shiny)
#install.packages("DT")
library(DT)

options(shiny.maxRequestSize = 100*1024^2)
ui <- fluidPage(
  titlePanel("Interactive Public Health Data Explorer"),
  sidebarLayout(
    sidebarPanel(
      fileInput("datafile", "Upload CSV only", accept = c(".csv")),
      conditionalPanel(
        condition = "input.tabs == 'Scatter Plot'",
        uiOutput("var_select")
      ),
      
      conditionalPanel(
        condition = "input.tabs == 'Diagnosis Filter'",
        uiOutput("diagnosis_column"),
        uiOutput("diagnosis_value_select")
      ),

      conditionalPanel(
        condition = "input.tabs == 'Summary Stats'",
        uiOutput("summary_var_select")
      ),
      conditionalPanel(
        condition = "input.tabs == 'Distribution'",
        uiOutput("dist_var_select")
      )
    ),
    mainPanel(
      tabsetPanel(
        id = "tabs",  # id to reference in conditionalPanel
        
        tabPanel("Preview", 
                 div(
                   style = "overflow-x: auto; width: 100%;",
                   tableOutput("preview")
                 )
                 
                 ),
        tabPanel("Scatter Plot", plotOutput("scatter")),
        tabPanel("Diagnosis Filter", 
                 div(#Using CSS 
                   style = "overflow-x: auto; width: 100%;",
                   tableOutput("diagnosis_table")
                 )

                 ),
        tabPanel("Summary Stats", verbatimTextOutput("summary")),
        tabPanel("Distribution", plotOutput("dist_plot"))
      )
    )
  )
)


server <- function(input, output, session) {
  # Reactive: Read uploaded data
  data <- reactive({
    req(input$datafile)
    read.csv(input$datafile$datapath, stringsAsFactors = FALSE)
  })
  
  # 1. Preview first 10 rows
  output$preview <-renderTable({
    req(data())
    head(data(), 10)
  })
  
  # 2. Dynamic variable selection for scatter plot
  output$var_select <- renderUI({
    req(data())
    num_vars <- names(data())[sapply(data(), is.numeric)]
    if (length(num_vars) < 2) return(NULL)
    tagList(
      selectInput("xvar", "X Variable", choices = num_vars),
      selectInput("yvar", "Y Variable", choices = num_vars, selected = num_vars[2])
    )
  })
  
  output$scatter <- renderPlot({
    req(data(), input$xvar, input$yvar)
    plot(data()[[input$xvar]], data()[[input$yvar]],
         xlab = input$xvar, ylab = input$yvar,
         main = paste("Scatterplot of", input$yvar, "vs", input$xvar),
         pch = 19, col = "steelblue")
  })
  
  # 3. Reactive filtering by diagnosis
  output$diagnosis_column <- renderUI({
    req(data())
    # Get all character columns
    char_cols <- c("Please Select",names(data())[sapply(data(), is.character)])
    #char_cols <- c("Please Select",char_cols[sapply(char_cols, function(col) length(unique(data()[[col]])) < 50)])

    # UI for selecting the diagnosis column
    tagList(
      selectInput("diagnosis_col", "Select Diagnosis Column", choices = char_cols),
      uiOutput("diagnosis_value_select")
    )
  })
  
  output$diagnosis_value_select <- renderUI({
    req(data(), input$diagnosis_col)
    # Get unique values from selected column
    choices <- unique(data()[[input$diagnosis_col]]) 
    selectInput("diagnosis", "Select Diagnosis", choices = choices)
  })
  
  output$diagnosis_table <- renderTable({
    req(data(), input$diagnosis_col, input$diagnosis)
    subset(data(), data()[[input$diagnosis_col]] == input$diagnosis)
  })
  
  
  # 4. Summary statistics display
  output$summary_var_select <- renderUI({
    req(data())
    num_vars <- names(data())[sapply(data(), is.numeric)]
    selectizeInput("summary_var", "Variable for Summary Stats", choices = num_vars)
  })
  
  output$summary <- renderPrint({
    req(data(), input$summary_var)
    x <- data()[[input$summary_var]]
    stats <- c(
      Mean = mean(x, na.rm = TRUE),
      Median = median(x, na.rm = TRUE),
      SD = sd(x, na.rm = TRUE)
    )
    print(stats)
  })
  
  # 5. Distribution panel
  output$dist_var_select <- renderUI({
    req(data())
    num_vars <- names(data())[sapply(data(), is.numeric)]
    selectInput("dist_var", "Variable for Distribution", choices = num_vars)
  })
  
  output$dist_plot <- renderPlot({
    req(data(), input$dist_var)
    hist(data()[[input$dist_var]], main = paste("Distribution of", input$dist_var),
         xlab = input$dist_var, col = "darkgreen", border = "white")
  })
}

shinyApp(ui, server)
# 
# server <- function(input, output, session) {
#   # Reactive expression to read the uploaded file
#   data <- reactive({
#     req(input$datafile)  # Wait until a file is uploaded
#     read.csv(input$datafile$datapath)
#   })
#   
#   # Output the number of rows
#   renderPrint({
#     nrow(data())
#   })
# }
# shinyApp(ui=ui,server)