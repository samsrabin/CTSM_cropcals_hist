%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Evaluating a CLM run with country-level FAO yields %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

calib_ver = 24 ;   % The version of mapping FAO to CLM crop types
ctrymapVer = 1 ;

% Compare individual years? Or take mean over period?
indiv_years = true ;

% Where are the CLM outputs saved?
clm0_file_yield = '/Users/Shared/CESM_runs/f10_f10_mg37_20220530/20220615.38a8cce.nativeCCs/f10_f10_mg37_20220530.nativeCCs_CropRemapped_GRAINC_TO_FOOD_198001-201012_Max.nc' ;
clm1_file_yield = '/Users/Shared/CESM_runs/f10_f10_mg37_20220530/20220615.38a8cce.gddforced/f10_f10_mg37_20220530_CropRemapped_GRAINC_TO_FOOD_198001-201012_Max.nc' ;

% What LU timeseries file was used for the CLM run?
lu_timeseries_file = '/Users/Shared/CESM_inputdata/lnd/clm2/surfdata_map/release-clm5.0.18/landuse.timeseries_10x15_hist_78pfts_CMIP6_simyr1850-2015_c190214.nc' ;

% What years are we looking at?
yearList = 1980:2010 ;

% CLM settings
harvest_efficiency = 0.85 ;
grain_cfrac = 0.45 ;


%% Setup

Nyears = length(yearList) ;

pftList = get_pftList() ;
cftList = pftList(15:end) ;
Ncft = length(cftList) ;

% bigCrops_fao = {'Maize', 'Wheat', 'Soybeans', 'Rice, paddy', 'Sugar cane', 'Seed cotton'} ;
% % bigCrops_fao = {'Maize', 'Wheat', 'Soybeans', 'Rice, paddy (rice milled equivalent)',  ...
% %     'Sugar cane', 'Seed cotton'} ;
% bigCrops_clm = {'corn', 'wheat', 'soy', 'rice', 'sugarcane', 'cotton'} ;
bigCrops_fao = {'Maize', 'Wheat', 'Soybeans', 'Rice, paddy'} ;
bigCrops_clm = {'corn', 'wheat', 'soy', 'rice'} ;
Nbig = length(bigCrops_clm) ;

clm_file_yield = clm1_file_yield ;


[prod_clm0_big_yc, area_clm0_big_yc, yield_clm0_big_yc, yield_clm0_big_y] = import_clm( ...
    clm0_file_yield, yearList, harvest_efficiency, grain_cfrac, lu_timeseries_file, ...
    Ncft, Nyears, Nbig, bigCrops_clm, cftList);

[prod_clm1_big_yc, area_clm1_big_yc, yield_clm1_big_yc, yield_clm1_big_y] = import_clm( ...
    clm1_file_yield, yearList, harvest_efficiency, grain_cfrac, lu_timeseries_file, ...
    Ncft, Nyears, Nbig, bigCrops_clm, cftList);



%% Import FAO data

fao = readtable('/Users/sam/Documents/git_repos/CTSM_myscripts/crop_calendar_MATLAB/FAOSTAT_data_6-15-2022.csv') ;
okYear = fao.Year >= min(yearList) & fao.Year <= max(yearList) ;
fao = fao(okYear,:) ;

for c = 1:Nbig
    thisCrop = bigCrops_fao{c} ;
    thisCrop_forArea = strrep(thisCrop, ' (rice milled equivalent)', '') ;
    if c==1
        prod_fao_big_yc = nan(size(prod_clm0_big_yc)) ;
        area_fao_big_yc = nan(size(area_clm0_big_yc)) ;
        yield_fao_big_yc = nan(size(prod_clm0_big_yc)) ;
    end
    prod_fao_big_yc(:,c) = fao.Value(strcmp(fao.Item, thisCrop) & strcmp(fao.Element, 'Production')) ;
    area_fao_big_yc(:,c) = fao.Value(strcmp(fao.Item, thisCrop_forArea) & strcmp(fao.Element, 'Area harvested')) ;
    yield_fao_big_yc(:,c) = fao.Value(strcmp(fao.Item, thisCrop_forArea) & strcmp(fao.Element, 'Yield')) ;
end

% Convert Rice yield to milled equivalent
if any(contains(bigCrops_fao, 'rice milled equivalent'))
    rice_ratio_y = fao.Value(strcmp(fao.Item, 'Rice, paddy') & strcmp(fao.Element, 'Production')) ...
        ./ fao.Value(strcmp(fao.Item, 'Rice, paddy (rice milled equivalent)') & strcmp(fao.Element, 'Production')) ;
    yield_fao_big_yc = yield_fao_big_yc ./ repmat(rice_ratio_y, [1 Nbig]) ;
end

% Convert from hg/ha to tons/ha
yield_fao_big_yc = yield_fao_big_yc * 1e-4 ;

% Get totals
yield_fao_big_y = sum(prod_fao_big_yc,2) ./ sum(area_fao_big_yc,2) ;

yield_fao_big_yc
yield_fao_big_y 


%% Plot each crop

lineWidth = 3 ;

figure('Color', 'w', 'Position', figurePos) ;
plot(yearList, yield_clm_big_yc, '-', ...
    'LineWidth', lineWidth)
hold on
set(gca, 'ColorOrderIndex', 1) ;
plot(yearList, yield_fao_big_yc, '--', ...
    'LineWidth', lineWidth)
hold off
legend(bigCrops_fao)
set(gca, 'FontSize', 18)


%% Plot totals

lineWidth = 3 ;

figure('Color', 'w', 'Position', [1   366   868   439]) ;
plot(yearList, yield_clm0_big_y, '-r', ...
    'LineWidth', lineWidth)
hold on
plot(yearList, yield_clm1_big_y, '-b', ...
    'LineWidth', lineWidth)
plot(yearList, yield_fao_big_y, '-k', ...
    'LineWidth', lineWidth)
hold off
legend({'CLM (original)', 'CLM (forced w/ GGCMI3', 'FAO'}, ...
    'Location','northwest')
set(gca, 'Box', 'off')
legend boxoff
% ylabel('Yield (tons ha^{-1})')
set(gca, 'FontSize', 24)

export_fig('/Users/Shared/CESM_runs/f10_f10_mg37_20220530/20220615.38a8cce/global_yield.pdf')


%% FUNCTIONS
function pft_list = get_pftList() 

pft_list = ...
    {'needleleaf_evergreen_temperate_tree','needleleaf_evergreen_boreal_tree', ...
    'needleleaf_deciduous_boreal_tree','broadleaf_evergreen_tropical_tree', ...
    'broadleaf_evergreen_temperate_tree','broadleaf_deciduous_tropical_tree', ...
    'broadleaf_deciduous_temperate_tree','broadleaf_deciduous_boreal_tree', ...
    'broadleaf_evergreen_shrub','broadleaf_deciduous_temperate_shrub', ...
    'broadleaf_deciduous_boreal_shrub','c3_arctic_grass','c3_non-arctic_grass','c4_grass', ...
    'unmanaged_c3_crop','unmanaged_c3_irrigated','temperate_corn','irrigated_temperate_corn', ...
    'spring_wheat','irrigated_spring_wheat','winter_wheat','irrigated_winter_wheat','soybean', ...
    'irrigated_soybean','barley','irrigated_barley','winter_barley','irrigated_winter_barley', ...
    'rye','irrigated_rye','winter_rye','irrigated_winter_rye','cassava','irrigated_cassava', ...
    'citrus','irrigated_citrus','cocoa','irrigated_cocoa','coffee','irrigated_coffee','cotton', ...
    'irrigated_cotton','datepalm','irrigated_datepalm','foddergrass','irrigated_foddergrass', ...
    'grapes','irrigated_grapes','groundnuts','irrigated_groundnuts','millet','irrigated_millet', ...
    'oilpalm','irrigated_oilpalm','potatoes','irrigated_potatoes','pulses','irrigated_pulses', ...
    'rapeseed','irrigated_rapeseed','rice','irrigated_rice','sorghum','irrigated_sorghum', ...
    'sugarbeet','irrigated_sugarbeet','sugarcane','irrigated_sugarcane','sunflower', ...
    'irrigated_sunflower','miscanthus','irrigated_miscanthus','switchgrass', ...
    'irrigated_switchgrass','tropical_corn','irrigated_tropical_corn','tropical_soybean', ...
    'irrigated_tropical_soybean'} ;
end


function [prod_clm_big_yc, area_clm_big_yc, yield_clm_big_yc, yield_clm_big_y] = import_clm( ...
    clm_file_yield, yearList, harvest_efficiency, grain_cfrac, lu_timeseries_file, ...
    Ncft, Nyears, Nbig, bigCrops_clm, cftList)

% Get CLM production

% Import yield

% yield_info = ncinfo(yield_file) ;
% yield_dims = {yield_info.Dimensions.Name} ;
yield_YXyc = permute(ncread(clm_file_yield, 'GRAINC_TO_FOOD'), [2 1 4 3]) ;
yearList_in = ncread(clm_file_yield, 'time') ;
[~,IA] = intersect(yearList_in, yearList) ;
yield_YXyc = yield_YXyc(:,:,IA,:) ;

% Convert to tons DM per ha
yield_YXyc = yield_YXyc * (harvest_efficiency / grain_cfrac) * 0.01 ;

% % Import crop areas
% cellarea_YX = permute(ncread(lu_timeseries_file, 'AREA'), [2 1]) ;
% landfrac_YX = permute(ncread(lu_timeseries_file, 'LANDFRAC_PFT'), [2 1]) ;
% cropfrac_file = strrep(yield_file, '.nc', 'FracArea.nc') ;
% cropfrac_YXyc = permute(ncread(cropfrac_file, 'fracarea'), [2 1 4 3]) ;
% 
% % Get production
% prod_clm_YXyc = yield_YXyc .* cropfrac_YXyc ;

% Import crop areas
cellarea_YX = permute(ncread(lu_timeseries_file, 'AREA'), [2 1]) ;
landfrac_YX = permute(ncread(lu_timeseries_file, 'LANDFRAC_PFT'), [2 1]) ;
landarea_YX = cellarea_YX .* landfrac_YX ;
cropfrac_YXy = permute(ncread(lu_timeseries_file, 'PCT_CROP'), [2 1 3]) / 100;
[~, IA] = intersect(ncread(lu_timeseries_file, 'time'), yearList) ;
cropfrac_YXy = cropfrac_YXy(:,:,IA) ;
cftfrac_YXyc = permute(ncread(lu_timeseries_file, 'PCT_CFT'), [2 1 4 3]) / 100;
cftfrac_YXyc = cftfrac_YXyc(:,:,IA,:) ;
cftarea_YXyc = cftfrac_YXyc .* repmat(cropfrac_YXy, [1 1 1 Ncft]) ...
    .* repmat(landarea_YX, [1 1 Nyears Ncft]) ;
% Convert from km2 to ha
cftarea_YXyc = cftarea_YXyc * 100 ;

% Get production
try
    prod_clm_YXyc = yield_YXyc .* cftarea_YXyc ;
catch ME
    keyboard
end


% % Remove crops with no area
% cropfrac_c = squeeze(sum(cropfrac_YXyc, 1:3, 'omitnan')) ;
% not_simd = cropfrac_c==0 ;
% yield_YXyc(:,:,:,not_simd) = [] ;
% cropfrac_YXyc(:,:,:,not_simd) = [] ;
% cftList(not_simd) = [] ;


% Get CLM production and area for Big 4

for c = 1:Nbig
    thisCrop = bigCrops_clm{c} ;
    if c==1
        prod_clm_big_YXyc = nan([size(prod_clm_YXyc, 1:3) Nbig]) ;
        area_clm_big_YXyc = nan([size(prod_clm_YXyc, 1:3) Nbig]) ;
    end
    isThisCrop = contains(cftList, thisCrop) ;
    if ~any(isThisCrop)
        error('No %s crops found', thisCrop)
    end
    prod_clm_big_YXyc(:,:,:,c) = sum(prod_clm_YXyc(:,:,:,isThisCrop), 4, 'omitnan') ;
    area_clm_big_YXyc(:,:,:,c) = sum(cftarea_YXyc(:,:,:,isThisCrop), 4, 'omitnan') ;
end

prod_clm_big_yc = squeeze(sum(prod_clm_big_YXyc, 1:2, 'omitnan')) ;
area_clm_big_yc = squeeze(sum(area_clm_big_YXyc, 1:2, 'omitnan')) ;
yield_clm_big_yc = prod_clm_big_yc ./ area_clm_big_yc
yield_clm_big_y = sum(prod_clm_big_yc,2) ./ sum(area_clm_big_yc,2)
end