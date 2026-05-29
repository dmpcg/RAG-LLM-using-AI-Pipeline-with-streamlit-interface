# Excel Model Research — Real-Model Structural Patterns

> Renamed from `findings.md` (WS-6 / P4-3). Source material: structural/naming patterns observed across 20+ real financial models. Drives future `excel_processor.py` and `line_item_mapper.py` work.

## RAG Pipeline Enhancement - Research Findings

## Real Model Analysis (20+ files from user's computer)

### Key Structural Patterns Discovered

1. **Row-based layouts dominate** - Most real models have line items as rows, periods as columns
2. **Labels are often in Column B** (not A) - Column A is sometimes blank or has section groupings
3. **Mixed section headers and data** - "Income Statement" appears as a row label mid-sheet
4. **Multi-entity models** - Separate sheets per entity (Gulf Shore, JSJ, River Road, Staffing)
5. **CSV exports lose headers** - Many CSVs have "Unnamed: 0" columns (Excel->CSV without header fix)
6. **Highly specific expense names** - "Utility-Electric (DTE)", "Lab Testing (Prism(formula))"
7. **Parenthetical qualifiers** - "(formula)", "(see below)", "(via credit card)", "(15th mthly)"
8. **280E tax classification** - Cannabis-specific: separate 280E classification sheet
9. **Weekly granularity** - Cash forecasts at weekly level, not monthly/quarterly
10. **Multi-bank tracking** - Comerica, Dart separate bank account tracking

### Cannabis-Specific Line Items Found
- Wholesale Revenue - Bigs/Smalls/Trim (product categories by size)
- Cost per LB, Realized Price/LB
- Seeds/Clones, Nutrients, IPM, Soil & Cubes
- Metrc Tags (regulatory tracking system)
- CRA Fines, License Renewal (Commercial/Medical, City/State)
- 280E Tax Rate vs Normalized Tax Rate
- Production Schedule, Yield, Incremental Yield from CAPEX
- Environmentals, Fertigation/Irrigation, Lights (CAPEX categories)

### LBO/Valuation Line Items Found
- Enterprise Value, Purchase Multiple, Exit Multiple
- Senior Debt Rate, Mezz Debt Rate, Leverage Ratio
- IRR, MOIC, Exit Value, Payback Period
- Hold Period, Discount Rate
- TTM Revenue, TTM EBITDA
- Unlevered FCF, Levered FCF, Equity FCF
- DSCR, Interest Coverage, Debt Balance
- Debt Service (senior_beginning/interest/principal/ending)

### Cash Forecast Line Items Found
- Beg Cash, AR Collections, Forecasted AR
- Total Cash Inflow, Total Cash Revenues
- Weekly CF FCST - Ending
- Payroll and Related Payroll Taxes
- Contract Labor, Temp Labor, Sales Commissions
- Workers Comp (quarterly), Employee Retention Credit
- Outside Broker Commission, Lab Testing
- Alarm System, Internet, E-Commerce
- Property Tax (Personal/Real Property)
- Debt & Interest (Financing CF)
- Cap Ex (Investing CF)

### KPI Dashboard Line Items Found
- Revenue per SqFt, Daily Transaction Count
- Average Transaction Value, Average Items per Transaction
- COGS % by location, COGS improvement targets (90d/180d/365d)
- Adult Use %, Medical Revenue
- Inventory Days on Hand, Cash Runway (months)
- Break-even analysis, Insolvency Date
- New vs Repeat Customer %
- Capacity Utilization
- Cash-on-Cash (Yr1), Annualized Cash Yield

### Debt Waterfall Line Items Found
- instrument_id, instrument_name, issuer_entity
- instrument_type, security_status, ranking
- principal_at_inception, principal_outstanding
- net_carrying_value, effective_interest_rate
- conversion_status, make-whole provisions
- fundamental_change_put, holder_put_right
- cross_default_provisions, financial_covenants

### Fund Reporting Line Items Found
- Month to Date, Year to Date, Inception to Date
- Beginning/Ending Capital Balance
- Contributions, Distributions, Net Cash Flow

### Construction Budget Line Items Found
- Code, Description, Original Budget, Change, Current Budget
- In Process, Payments, Remaining, Complete %

## Non-Standard Naming Patterns

### Abbreviation Patterns
- D&A, SG&A, G&A, A/R, A/P, NWC, DSCR, CCC
- GSH (Gulf Shore), JSJ, IPP (Inkster Pine Park)
- DTE (Detroit Edison - utility provider)
- CRA (Cannabis Regulatory Agency)

### Parenthetical Context
- "(formula)" - calculated field
- "(see below)" - reference to detail section
- "(via credit card)" - payment method
- "(15th mthly)" - payment timing
- "(27th Qtly Mar/Jun/Sept/Dec)" - quarterly payment schedule
- "(inc commissions0)" - typo in real model

### Section Header Patterns
- "Income Statement" / "Cash Flow Statement" as row labels
- "Expenses:" as a section divider (with colon)
- "Cost of Goods Sold- Direct" / "Cost of Goods Sold- Indirect" (with dash)
- "$" and "(%)" as column sub-headers for units

### Time Period Patterns Found
- "Week Number", "Year Number"
- "YTD Feb", "MTD", "QTD"
- "2024A" (Actual), "2025E" (Estimate)
- "Month to Date", "Year to Date", "Inception to Date"
- "8 Weeks Forecast thru 12/26/25"
- "Actual / Forecast" toggle row
