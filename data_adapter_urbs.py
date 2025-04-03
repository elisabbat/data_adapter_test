import os
import shutil
import urbs
import dataclasses
import pandas as pd

from data_adapter import databus, collection, main
from data_adapter.preprocessing import Adapter
from data_adapter.structure import Structure

#class names
#define functions
#name objects
standard_units = {
    'None': None,
    'year': 'a',
    'cost': 'MEUR',
    'energy': 'GWh',
    'power': 'GW',
    'efficiency': '%',
    'transport_pass_demand': 'Gpkm',
    'vehicles': 'kvehicles',
    'emissions': 'Mt',
    'pass_transport_ccf': 'Gpkm/kvehicles',
    'energy_transport_ccf': 'GWh/kvehicles',
    'power_per_vehicle': 'GW/kvehicles',
    'milage': 'Tm/(kvehicles*a)',
    'self_discharge': '%/h',
    'cost_per_capacity': 'MEUR/GW',
    'cost_per_energy': 'MEUR/GWh',
    'cost_per_vehicle': 'MEUR/kvehicles',
    'cost_per_pkm': 'MEUR/Gpkm',
    'cost_var_per_vehicle': 'MEUR/(kvehicles*a)',
    'specific_emission': 'Mt/GWh',
    'specific_emission_co2': 'MtCO2/GWh',
    'ccf_vehicles': 'GWh/100km',
    # 'misc_ccf': 'MWh/MWh',
    'misc_ts': 'kW/kW',
    'occupancy_rate': 'persons/vehicle',
}
class Datapackage: #from data_adapter_oemof build_datapackage
    parametrized_elements: dict[
        str, pd.DataFrame()
    ]  # datadict with scalar data in form of {type:pd.DataFrame(type)}
    parametrized_sequences: dict[
        str, pd.DataFrame()
    ]  # timeseries in form of {type:pd.DataFrame(type)}
    foreign_keys: dict  # foreign keys for timeseries profiles
    adapter: Adapter
    periods: pd.DataFrame()
    location_to_save_to: str = None
    tsa_parameters: pd.DataFrame = None

@dataclasses.dataclass
class DataAdapter:
    def __init__(
            self,
            url: str,
            structure_name: str,
            process_sheet: str,
            helper_sheet: str,
            downloadData: bool = True,
    ):
        self.op_flex_opt = None
        if downloadData:
            databus.download_collection(url)
        self.collection_name = url.split('/')[-1]
        self.scenario = 'f_tra_tokio'
        self.bev_constraint_data = {}
        self.structure = Structure(
            structure_name,
            process_sheet=process_sheet,
            helper_sheet=helper_sheet,
        )
        self.adapter = Adapter(
            self.collection_name,
            structure=self.structure,
            #units=list(utils.standard_units.values()),
        )

    def export_results(self): #already the end result,skip
        results_df = pd.DataFrame()
        results_df = results_df.reindex(
            columns=['scenario', 'year', 'process', 'parameter', 'sector', 'category', 'specification', 'groups',
                     'new', 'input_groups', 'output_groups', 'unit', 'value']
        )
        results_df.loc[
            (results_df['unit'] == standard_units['power']) & (results_df['parameter'] == 'flow_volume'), 'unit'
        ] = standard_units['energy']
        if os.getcwd().endswith('data_adapter_urbs'):
            results_df.to_csv('examples/output/test_results.csv', index=False, sep=';')
        else:
            results_df.to_csv('results/test_results.csv', index=False, sep=';')

        return results_df

    def get_comp_results(self, comp_name, model_results):
        # set descriptive data for component
        comp_df = pd.DataFrame()
        comp_results_base = {
            'scenario': [self.scenario],
            'process': [comp_name],
        }
        #comp = self.esM.getComponent(comp_name)
        comp_name_split = comp_name.split('_')
        if len(comp_name_split) > 1:
            comp_results_base['sector'] = [comp_name_split[0]]
            comp_results_base['category'] = [comp_name_split[1]]
            comp_results_base['specification'] = [comp_name_split[2:-1]]
        if comp_name[-1] == '1':
            comp_results_base['new'] = 1
        elif comp_name[-1] == '0':
            comp_results_base['new'] = 0
        else:
            comp_results_base['new'] = 0  # TODO check components and define value

da = DataAdapter
url = main.download_collection()
da(url,'SEDOS_Modellstruktur_steel_industry','Process_Set','Helper_Set')


"""def test_return():
    adapter_results = "url"
    return adapter_results"""
