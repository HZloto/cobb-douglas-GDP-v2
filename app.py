#!/usr/bin/env python3
# filepath: /Users/hgoz/Documents/WS Anal/cobb-douglas-GDP-v2/app.py

import os
import json
import logging
import pandas as pd
from typing import Optional, Dict, Any

# Import our modules
from Agent import generate
from main import Settings, run_regional_analysis

def get_policy_scenario(prompt: str) -> Optional[Dict[str, Any]]:
    """
    Use the Agent to generate a policy scenario based on the provided prompt.
    
    Args:
        prompt: A string describing the policy action to analyze
        
    Returns:
        A dictionary containing the policy scenario or None if an error occurs
    """
    try:
        # The generate function returns a JSON string with the policy analysis
        # Since it's designed for print output, we'll capture it
        import io
        import sys
        
        # Redirect stdout to capture the output
        original_stdout = sys.stdout
        sys.stdout = string_io = io.StringIO()
        
        # Call the generate function
        generate(prompt)
        
        # Get the output and restore stdout
        json_output = string_io.getvalue()
        sys.stdout = original_stdout
        
        # Parse the JSON output
        policy_scenario = json.loads(json_output)
        
        logging.info(f"Generated policy scenario for KPI: {policy_scenario.get('KPI')}, "
                     f"region: {policy_scenario.get('region')}, "
                     f"sector: {policy_scenario.get('sector')}")
        
        return policy_scenario
    except Exception as e:
        logging.error(f"Failed to generate policy scenario: {e}")
        return None

def analyze_policy_impact(prompt: str, save_results: bool = True) -> Optional[pd.DataFrame]:
    """
    Analyze the impact of a policy described in the prompt by:
    1. Generating a policy scenario using the Agent
    2. Running the GDP analysis with the policy scenario
    
    Args:
        prompt: A string describing the policy action to analyze
        save_results: Whether to save the results to a file
        
    Returns:
        A DataFrame with the analysis results or None if an error occurs
    """
    try:
        # 1. Get policy scenario from the prompt
        policy_scenario = get_policy_scenario(prompt)
        if policy_scenario is None:
            logging.error("Could not generate policy scenario from prompt")
            return None
            
        # 2. Configure the analysis
        cfg = Settings()
        
        # Change output filename to include policy info before running analysis
        kpi = policy_scenario.get("KPI", "unknown")
        region = policy_scenario.get("region", "ALL")
        sector = policy_scenario.get("sector", "ALL")
        
        # Create a new filename for the output
        policy_output_filename = f"consolidated_gdp_policy_{kpi}_{region}_{sector}.csv"
        
        # 3. Run the analysis with the policy scenario, passing the custom filename
        results = run_regional_analysis(
            cfg,
            save_intermediate=False, 
            generate_tracking=False,
            policy_scenario=policy_scenario,
            output_filename=policy_output_filename
        )
        

        # 4. Save or display some summary stats
        if results is not None and save_results:
            # Create a summary of results for quick reference
            summary = results[results['row_type'] == 'grand_total'].copy()
            summary_file = os.path.join(cfg.output_dir, f"policy_impact_summary_{kpi}_{region}_{sector}.csv")
            summary.to_csv(summary_file, index=False)
            logging.info(f"Summary results saved to {summary_file}")
            
            # Also save baseline for comparison
            baseline_cfg = Settings()
            baseline_results = run_regional_analysis(
                baseline_cfg,
                save_intermediate=False,
                generate_tracking=False,
                policy_scenario=None
            )
            
            if baseline_results is not None:
                baseline_summary = baseline_results[baseline_results['row_type'] == 'grand_total'].copy()
                baseline_file = os.path.join(cfg.output_dir, "consolidated_gdp_baseline.csv")
                baseline_results.to_csv(baseline_file, index=False)
                logging.info(f"Baseline results saved to {baseline_file}")
                
                # Calculate percentage difference
                comparison = pd.merge(
                    summary, 
                    baseline_summary,
                    on=['year', 'regioncode', 'sector', 'row_type'],
                    suffixes=('_policy', '_baseline')
                )
                comparison['pct_diff'] = ((comparison['gdp_policy'] / comparison['gdp_baseline']) - 1) * 100
                comparison_file = os.path.join(cfg.output_dir, f"policy_impact_diff_{kpi}_{region}_{sector}.csv")
                comparison.to_csv(comparison_file, index=False)
                logging.info(f"Policy vs baseline comparison saved to {comparison_file}")
        
        return results
    
    except Exception as e:
        logging.error(f"Error analyzing policy impact: {e}")
        return None

def main():
    """Main CLI interface for the policy impact analysis tool."""
    print("\n*** Saudi Arabia Policy Impact Analysis Tool ***\n")
    print("This tool analyzes the economic impact of policy actions on GDP components.")
    print("Enter a description of a policy action, and the tool will:")
    print(" 1. Generate a policy scenario using AI analysis")
    print(" 2. Calculate the impact on GDP using a Cobb-Douglas model")
    print(" 3. Save the results for comparison with baseline\n")
    
    while True:
        # Get policy description from user
        print("\nEnter policy description (or 'quit' to exit):")
        prompt = input("> ")
        
        if prompt.lower() in ('quit', 'exit', 'q'):
            break
            
        if not prompt.strip():
            print("Please enter a policy description.")
            continue
            
        print("\nAnalyzing policy impact. This may take a moment...\n")
        
        # Run analysis
        results = analyze_policy_impact(prompt)
        
        if results is not None:
            # Show a summary
            summary = results[results['row_type'] == 'grand_total'].copy()
            print("\nGDP Impact Summary:")
            print(summary[['year', 'gdp']].to_string(index=False))
            
            print("\nAnalysis complete! Results saved to outputs directory.")
            print("Would you like to analyze another policy? (y/n)")
            if input("> ").lower() != 'y':
                break
        else:
            print("\nAnalysis failed. Please try again with a different prompt.")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Run the main CLI interface
    main()
