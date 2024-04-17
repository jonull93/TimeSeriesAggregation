from TSA.main import from_df

# If termcolor is not installed, prompt the user that it's a nice-to-have
try:
    import termcolor
except ImportError:
    print("It is recommended to install termcolor for better colored warnings etc.")