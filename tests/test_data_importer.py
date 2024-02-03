import unittest


class TestDataImporter(unittest.TestCase):
    def test_csv_import(self):
        # Test importing CSV files
        importer = DataImporter("path/to/csv")
        data = importer.load()
        self.assertIsNotNone(data)  # basic check to ensure data is not None

if __name__ == '__main__':
    unittest.main()
