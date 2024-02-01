# Project Description: Time Series Aggregation Tool

### Aim

The primary aim of this project is to develop a Python-based tool for aggregating time-series data. This tool will initially focus on implementing the Priority Chronological Time-Period Clustering (PCTPC) method, with the flexibility to integrate additional methods in the future. The tool will be designed to handle various types of time series data, such as electricity load and wind/PV generation profiles, and aggregate them over reduced timesteps.

Where appropriate, the code will be object-oriented because:
* I want to learn OOP
* It seems suitable for modularity and the option of adding features over time without spaghettifying the code
* More maintainable?

### Scope

* **Data Handling**: Import, process, and manage time series data from various sources and formats.
* **Algorithm Implementation**: Implement the PCTPC method as the core aggregation algorithm, with down-sampling as a dummy comparison.
* **User Interaction**: Provide a user-friendly command-line interface (CLI) to allow users to specify input data, choose aggregation methods, and set relevant parameters.
* **Output Generation**: Output the aggregated time series data in a format suitable for plug-and-play into models as well as further analysis or visualization. When using PCTPC, multiple outputs will be created at varying levels of aggregation - each paired with simple figures to illustrate the level of aggregation.
 
## Project structure

### Proposal 1:
* `data_importer.py`: Handles importing and preprocessing of data.
* `config_manager.py`: Manages user inputs and configuration settings.
* `aggregation_algorithms/`: A directory for your algorithms.
    * `base.py`: Contains the base class for aggregation algorithms.
    * `pctpc.py`: The PCTPC algorithm implementation.
    * `other_algorithms.py`: Placeholders for future algorithms.
* `data_processor.py`: Performs data transformations.
* `data_exporter.py`: Manages the output of aggregated data.
* `main.py`: The main script where the workflow is defined.
* `tests/`: A directory for your test cases.
    * `test_data_importer.py`
    * `test_aggregation_algorithms.py`
    * `etc.`

### Features to include

#### 1\. **Data Import and Handling**

Reading inputs in the form of timeseries and, optionally, a capacity mix to calculate net-load.
* **Features**:
    * Support for multiple file formats (CSV, Excel, JSON, etc.).
    * Handling different time resolutions and formats.
    * Managing missing or inconsistent data?
* **Code Structure**:
    * A `DataImporter` class with methods for loading data from different sources.
    * Functions for data validation and preprocessing.

#### 2\. **User Interaction and Configuration**

* **Features**:
    * A user "interface" for setting up analysis parameters.
    * Ability to choose aggregation methods and adjust their settings.
* **Code Structure**:
    * A `ConfigurationManager` class to handle user inputs and settings.
    * Command Line Interface (CLI) or Graphical User Interface (GUI) components.

#### 3\. **Aggregation Algorithms**

* **Features**:
    * Implementation of PCTPC method.
    * Framework to add new aggregation methods.
    * Adjustable parameters for each method.
* **Code Structure**:
    * An abstract `AggregationAlgorithm` class defining a standard interface.
    * Derived classes like `PCTPCAggregator` implementing specific algorithms.
    * A factory or strategy pattern to instantiate algorithm objects dynamically.

#### 4\. **Data Processing and "Transformation"**

* **Features**:
    * Operations like normalization, scaling, or custom transformations.
    * Handling time-series specifics like seasonality or trends.
* **Code Structure**:
    * A `DataProcessor` class with methods for various transformations.
    * Integration with the aggregation modules to apply transformations as needed.

#### 5\. **Output Generation and Export**

* **Features**:
    * Generating aggregated time series.
    * Exporting results in different formats.
    * Visualizations of aggregated data.
* **Code Structure**:
    * A `DataExporter` class for saving results.
    * Functions or classes for data visualization and reporting.

#### 6\. **Documentation and Help**

* **Features**:
    * In-line code documentation.
    * User manual or guide.
    * Examples and tutorials.
* **Code Structure**:
    * Comprehensive comments and docstrings within the code.
    * Separate documentation files or an integrated help system.

#### 7\. **Testing and Validation**

* **Features**:
    * Unit tests for individual modules.
    * Integration tests for overall workflow.
    * Performance benchmarks.
* **Code Structure**:
    * A `TestSuite` with a set of automated tests.
    * Integration with a continuous integration (CI) system for automated testing.