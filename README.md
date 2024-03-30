# data_checklist

Example command to run the applicability and viability tests on SNLI, with premise-hypothesis overlap as the feature to test for applicability ("regular_vinfo" is the test for viability):

`accelerate launch main.py check_types=[applicability,regular_vinfo] std_transform_func=std_nli attribute_func=[snli_overlap] num_train_epochs=2 batch_size=8 +data=snli +model=t5-base`

To replicate the experiments in the paper on tbe length attribute of SHP:

`accelerate launch main.py check_types=[applicability,sufficiency,exclusivity,necessity] std_transform_func=std_shp null_transform_func=null_shp attribute_func=[shp_word_length] inverse_attribute_func=[inv_shp_word_length] batch_size=8 +data=shp +model=t5-base`


To add your own dataset, create a .yaml file under `configs/data`.

For a list of currently implemented data transforms that extract various attributes, see `src/data_transforms.py`. Additional data transforms can be added by implementing a new method in the `InputTransforms` class and adding it to the `self.transform_name_to_func` dict.


