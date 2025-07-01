import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from mordred import Calculator, descriptors
from tqdm.auto import tqdm
import logging
import sys
import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ADMETLipinskiScreener:
    def __init__(self):
        """Initialize with all necessary calculators"""
        self.lipinski_filters = self._setup_lipinski_filters()
        self.admet_calculators = self._setup_admet_calculators()
        
    def _setup_lipinski_filters(self):
        """Initialize Lipinski's Rule of Five filters with fallback"""
        try:
            params = FilterCatalogParams()
            # Try all known Lipinski filter names
            for name in ['LIPINSKI', 'LIPINSKI_ROF', 'LIPINSKI_RULEOF5']:
                try:
                    catalog = getattr(FilterCatalogParams.FilterCatalogs, name)
                    params.AddCatalog(catalog)
                    logger.info(f"Using built-in Lipinski filter: {name}")
                    return FilterCatalog(params)
                except AttributeError:
                    continue
            logger.warning("No built-in Lipinski filter found - using manual implementation")
            return None
        except Exception as e:
            logger.error(f"FilterCatalog initialization failed: {str(e)}")
            return None
    
    def _setup_admet_calculators(self):
        """Initialize ADMET property calculators"""
        return {
            'basic': self._calculate_basic_properties,
            'druglike': self._calculate_druglikeness,
            'tox': self._calculate_toxicity_risks,
            'pharma': self._calculate_pharmacokinetics
        }
    
    def _calculate_basic_properties(self, mol):
        """Calculate fundamental molecular properties"""
        return {
            'mw': Descriptors.MolWt(mol),
            'logp': Descriptors.MolLogP(mol),
            'hbd': Lipinski.NumHDonors(mol),
            'hba': Lipinski.NumHAcceptors(mol),
            'rotatable_bonds': Lipinski.NumRotatableBonds(mol),
            'tpsa': Descriptors.TPSA(mol),
            'n_rings': Lipinski.RingCount(mol),
            'aromatic_rings': Lipinski.NumAromaticRings(mol),
            'heavy_atoms': Lipinski.HeavyAtomCount(mol),
            'fraction_csp3': Lipinski.FractionCSP3(mol),
            'chiral_centers': len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
        }
    
    def _calculate_druglikeness(self, mol):
        """Calculate drug-like properties"""
        return {
            'lipinski_pass': self.check_lipinski(mol),
            'ghose_pass': self.check_ghose(mol),
            'veber_pass': self.check_veber(mol),
            'qed_score': self.calculate_qed(mol),
            'n_violations': self.count_violations(mol)
        }
    
    def _calculate_toxicity_risks(self, mol):
        """Calculate toxicity risk indicators"""
        return {
            'pains': self.check_pains(mol),
            'brenk': self.check_brenk(mol),
            'mutagenic_alert': self.check_mutagenic_alerts(mol),
            'synthetic_accessibility': self.estimate_synthetic_accessibility(mol)
        }
    
    def _calculate_pharmacokinetics(self, mol):
        """Estimate pharmacokinetic properties"""
        return {
            'bioavailability_score': self.calculate_bioavailability_score(mol),
            'pampa_permeability': self.estimate_pampa_permeability(mol),
            'cyp_inhibition_risk': self.estimate_cyp_inhibition(mol),
            'pgp_substrate_prob': self.estimate_pgp_substrate(mol)
        }
    
    def check_lipinski(self, mol):
        """Check compliance with Lipinski's Rule of Five"""
        if not mol:
            return False
        
        # Standard Rule of Five criteria
        conditions = [
            Descriptors.MolWt(mol) <= 500,
            Descriptors.MolLogP(mol) <= 5,
            Lipinski.NumHDonors(mol) <= 5,
            Lipinski.NumHAcceptors(mol) <= 10,
            Lipinski.NumRotatableBonds(mol) <= 10
        ]
        return all(conditions)
    
    def check_ghose(self, mol):
        """Check Ghose filter criteria"""
        try:
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            n_atoms = Lipinski.HeavyAtomCount(mol)
            conditions = [
                160 <= mw <= 480,
                -0.4 <= logp <= 5.6,
                20 <= n_atoms <= 70
            ]
            return all(conditions)
        except:
            return False
    
    def check_veber(self, mol):
        """Check Veber's rules for oral bioavailability"""
        try:
            conditions = [
                Lipinski.NumRotatableBonds(mol) <= 10,
                Descriptors.TPSA(mol) <= 140
            ]
            return all(conditions)
        except:
            return False
    
    def calculate_qed(self, mol):
        """Calculate Quantitative Estimate of Drug-likeness (QED)"""
        try:
            from rdkit.Chem.QED import qed
            return qed(mol)
        except:
            return np.nan
    
    def count_violations(self, mol):
        """Count number of rule violations"""
        if self.lipinski_filters:
            return len(self.lipinski_filters.GetMatch(mol)) if self.lipinski_filters.HasMatch(mol) else 0
        return sum([
            not (Descriptors.MolWt(mol) <= 500),
            not (Descriptors.MolLogP(mol) <= 5),
            not (Lipinski.NumHDonors(mol) <= 5),
            not (Lipinski.NumHAcceptors(mol) <= 10)
        ])
    
    def check_pains(self, mol):
        """Check for PAINS (pan-assay interference compounds) alerts"""
        try:
            params = FilterCatalogParams()
            params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
            catalog = FilterCatalog(params)
            return catalog.HasMatch(mol)
        except:
            return False
    
    def check_brenk(self, mol):
        """Check for Brenk alerts (unwanted functional groups)"""
        try:
            params = FilterCatalogParams()
            params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
            catalog = FilterCatalog(params)
            return catalog.HasMatch(mol)
        except:
            return False
    
    def check_mutagenic_alerts(self, mol):
        """Check for mutagenic alerts"""
        try:
            from rdkit.Chem import rdMolDescriptors
            return rdMolDescriptors.CalcNumAlerts(mol, "Mutagenicity")
        except:
            return 0
    
    def estimate_synthetic_accessibility(self, mol):
        """Estimate synthetic accessibility score (placeholder)"""
        try:
            # This is a simplified approximation - consider using actual SAS implementation
            complexity = (
                0.1 * Lipinski.HeavyAtomCount(mol) +
                0.2 * Lipinski.RingCount(mol) +
                0.3 * len(Chem.FindMolChiralCenters(mol, includeUnassigned=True)) +
                0.4 * (5 - min(Descriptors.NumSaturatedCarbocycles(mol), 5)))
            return max(1, min(10, complexity))
        except:
            return np.nan
    
    def calculate_bioavailability_score(self, mol):
        """Calculate bioavailability score (simplified)"""
        try:
            score = 0
            # Molecular weight component
            mw = Descriptors.MolWt(mol)
            score += max(0, 1 - (mw - 300)/200) if mw > 300 else 1
            
            # LogP component
            logp = Descriptors.MolLogP(mol)
            score += max(0, 1 - abs(logp - 2)/3)
            
            # PSA component
            tpsa = Descriptors.TPSA(mol)
            score += max(0, 1 - (tpsa - 60)/100)
            
            # Rotatable bonds
            rot_bonds = Lipinski.NumRotatableBonds(mol)
            score += max(0, 1 - rot_bonds/8)
            
            return score / 4  # Normalize to 0-1 range
        except:
            return np.nan
    
    def estimate_pampa_permeability(self, mol):
        """Estimate PAMPA permeability (simplified model)"""
        try:
            logp = Descriptors.MolLogP(mol)
            tpsa = Descriptors.TPSA(mol)
            mw = Descriptors.MolWt(mol)
            
            # Simple permeability model
            perm_score = (0.5 * logp) - (0.003 * tpsa) - (0.002 * mw) + 2.5
            return 1 / (1 + np.exp(-perm_score))  # Sigmoid to get 0-1 probability
        except:
            return np.nan
    
    def estimate_cyp_inhibition(self, mol):
        """Estimate CYP inhibition risk (placeholder)"""
        try:
            from rdkit.Chem import rdMolDescriptors
            alerts = rdMolDescriptors.CalcNumAlerts(mol, "CYP450")
            return min(1, alerts * 0.2)  # Convert alert count to risk score
        except:
            return np.nan
    
    def estimate_pgp_substrate(self, mol):
        """Estimate P-gp substrate probability (simplified)"""
        try:
            tpsa = Descriptors.TPSA(mol)
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            
            # Simple model based on literature thresholds
            if tpsa > 90 and mw > 400 and logp > 2:
                return 0.8  # High probability
            elif tpsa > 70 and mw > 350:
                return 0.5  # Medium probability
            else:
                return 0.2  # Low probability
        except:
            return np.nan
    
    def standardize_molecule(self, mol):
        """Standardize molecule structure"""
        try:
            # Neutralize charges
            from rdkit.Chem.MolStandardize import rdMolStandardize
            uncharger = rdMolStandardize.Uncharger()
            mol = uncharger.uncharge(mol)
            
            # Remove salts and solvents
            from rdkit.Chem.SaltRemover import SaltRemover
            remover = SaltRemover()
            mol = remover.StripMol(mol)
            
            # Kekulize and sanitize
            Chem.Kekulize(mol)
            Chem.SanitizeMol(mol)
            
            return mol
        except:
            return None
    
    def process_molecule(self, smi):
        """Process a single molecule"""
        try:
            mol = Chem.MolFromSmiles(smi)
            if not mol:
                return None
                
            mol = self.standardize_molecule(mol)
            if not mol:
                return None
                
            props = {'smiles': smi}
            
            # Calculate all properties
            for prop_type, calculator in self.admet_calculators.items():
                try:
                    props.update(calculator(mol))
                except Exception as e:
                    logger.debug(f"Error calculating {prop_type} for {smi}: {str(e)}")
                    continue
            
            return props
        except Exception as e:
            logger.debug(f"Error processing {smi}: {str(e)}")
            return None
    
    def process_molecules(self, smiles_list, n_jobs=None):
        """Process molecules in parallel"""
        if n_jobs is None:
            n_jobs = cpu_count() - 1 if cpu_count() > 1 else 1
            
        logger.info(f"Processing {len(smiles_list)} molecules with {n_jobs} workers")
        
        with Pool(n_jobs) as pool:
            results = list(tqdm(
                pool.imap(self.process_molecule, smiles_list),
                total=len(smiles_list),
                desc="Screening molecules"
            ))
        
        # Filter out None results
        valid_results = [r for r in results if r is not None]
        logger.info(f"Successfully processed {len(valid_results)}/{len(smiles_list)} molecules")
        
        return pd.DataFrame(valid_results)
    
    def filter_molecules(self, df, filters=None):
        """Filter molecules based on criteria"""
        if filters is None:
            # Default strict drug-like filters
            filters = {
                'lipinski_pass': True,
                'ghose_pass': True,
                'veber_pass': True,
                'mw': (200, 500),
                'logp': (-1, 5),
                'tpsa': (20, 140),
                'rotatable_bonds': (0, 10),
                'n_violations': (0, 1),
                'qed_score': (0.5, 1.0),
                'pains': False,
                'brenk': False,
                'mutagenic_alert': 0,
                'synthetic_accessibility': (1, 6),
                'bioavailability_score': (0.5, 1.0),
                'pampa_permeability': (0.5, 1.0)
            }
        
        mask = pd.Series([True] * len(df))
        
        for col, val in filters.items():
            if col in df.columns:
                if isinstance(val, (list, tuple)):
                    mask &= (df[col] >= val[0]) & (df[col] <= val[1])
                else:
                    mask &= (df[col] == val)
        
        return df[mask].copy()

def main():
    """Example usage"""
    # Sample molecules (replace with your dataset)
    df = pd.read_excel('~/Desktop/Anush Karampuri/MODEL/BInary_method/Virtual Screening/docking_validations/glide-dock_SP_chembl_cryst_dock/new_excel.xlsx')
    df = df.drop_duplicates('smiles')
    sample_smiles = df['smiles']
    
    try:
        # Initialize screener
        screener = ADMETLipinskiScreener()
        
        # Process molecules
        results_df = screener.process_molecules(sample_smiles, n_jobs=20)
        
        if results_df.empty:
            logger.error("No molecules processed successfully")
            sys.exit(1)
            
        # Filter molecules
        filtered_df = screener.filter_molecules(results_df)
        
        # Save results
        results_df.to_csv('all_molecules_with_properties.csv', index=False)
        filtered_df.to_csv('filtered_druglike_molecules.csv', index=False)
        
        logger.info(f"\nResults summary:")
        logger.info(f"Total molecules processed: {len(results_df)}")
        logger.info(f"Drug-like molecules found: {len(filtered_df)}")
        logger.info(f"Filter pass rate: {len(filtered_df)/len(results_df):.1%}")
        
        # Show filtered molecules
        if not filtered_df.empty:
            logger.info("\nFiltered molecules:")
            print(filtered_df[['smiles', 'mw', 'logp', 'hbd', 'hba', 'qed_score', 'bioavailability_score']])
        
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
