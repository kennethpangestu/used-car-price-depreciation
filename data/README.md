# Data

`raw/` holds the "100,000 UK Used Car Data set": used car listings scraped in 2020, one CSV per brand. It was published on Kaggle by Aditya Desai; this copy of the original per-brand files was taken from the public GitHub repository [farahemadd/AutoWorth-AI](https://github.com/farahemadd/AutoWorth-AI).

| File | Brand | Rows |
|---|---|---|
| `audi.csv` | Audi | 10,668 |
| `bmw.csv` | BMW | 10,781 |
| `ford.csv` | Ford | 17,965 |
| `hyundi.csv` | Hyundai (file name misspelled in the source) | 4,860 |
| `merc.csv` | Mercedes-Benz | 13,119 |
| `skoda.csv` | Skoda | 6,267 |
| `toyota.csv` | Toyota | 6,738 |
| `vauxhall.csv` | Vauxhall | 13,632 |
| `vw.csv` | Volkswagen | 15,157 |

| Column | Description |
|---|---|
| `model` | Model name (e.g. 3 Series, Fiesta) |
| `year` | Registration year |
| `price` | Listing price in GBP |
| `transmission` | Manual, Automatic, Semi-Auto or Other |
| `mileage` | Miles driven |
| `fuelType` | Petrol, Diesel, Hybrid, Electric or Other |
| `tax` | Annual UK road tax in GBP (`tax(£)` in the Hyundai file) |
| `mpg` | Fuel economy, miles per gallon |
| `engineSize` | Engine size in litres |

Cleaning rules are in `src/data.py`.
