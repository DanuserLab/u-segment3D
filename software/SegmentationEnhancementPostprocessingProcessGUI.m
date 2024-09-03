function varargout = SegmentationEnhancementPostprocessingProcessGUI(varargin)
%SEGMENTATIONENHANCEMENTPOSTPROCESSINGPROCESSGUI MATLAB code file for SegmentationEnhancementPostprocessingProcessGUI.fig
%      SEGMENTATIONENHANCEMENTPOSTPROCESSINGPROCESSGUI, by itself, creates a new SEGMENTATIONENHANCEMENTPOSTPROCESSINGPROCESSGUI or raises the existing
%      singleton*.
%
%      H = SEGMENTATIONENHANCEMENTPOSTPROCESSINGPROCESSGUI returns the handle to a new SEGMENTATIONENHANCEMENTPOSTPROCESSINGPROCESSGUI or the handle to
%      the existing singleton*.
%
%      SEGMENTATIONENHANCEMENTPOSTPROCESSINGPROCESSGUI('Property','Value',...) creates a new SEGMENTATIONENHANCEMENTPOSTPROCESSINGPROCESSGUI using the
%      given property value pairs. Unrecognized properties are passed via
%      varargin to SegmentationEnhancementPostprocessingProcessGUI_OpeningFcn.  This calling syntax produces a
%      warning when there is an existing singleton*.
%
%      SEGMENTATIONENHANCEMENTPOSTPROCESSINGPROCESSGUI('CALLBACK') and SEGMENTATIONENHANCEMENTPOSTPROCESSINGPROCESSGUI('CALLBACK',hObject,...) call the
%      local function named CALLBACK in SEGMENTATIONENHANCEMENTPOSTPROCESSINGPROCESSGUI.M with the given input
%      arguments.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES
%
% Copyright (C) 2024, Danuser Lab - UTSouthwestern 
%
% This file is part of uSegment3D_Package.
% 
% uSegment3D_Package is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
% 
% uSegment3D_Package is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with uSegment3D_Package.  If not, see <http://www.gnu.org/licenses/>.
% 
% 

% Edit the above text to modify the response to help SegmentationEnhancementPostprocessingProcessGUI

% Last Modified by GUIDE v2.5 21-Aug-2024 13:09:07

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @SegmentationEnhancementPostprocessingProcessGUI_OpeningFcn, ...
                   'gui_OutputFcn',  @SegmentationEnhancementPostprocessingProcessGUI_OutputFcn, ...
                   'gui_LayoutFcn',  [], ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
   gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before SegmentationEnhancementPostprocessingProcessGUI is made visible.
function SegmentationEnhancementPostprocessingProcessGUI_OpeningFcn(hObject, eventdata, handles, varargin)

processGUI_OpeningFcn(hObject, eventdata, handles, varargin{:},'initChannel',1);

% Parameter setup
userData = get(handles.figure1, 'UserData');
if isempty(userData), userData = struct(); end
funParams = userData.crtProc.funParams_;


% set GUI with Parameters:

% "Diffusion" panel. tag names pattern as type_d_parameterName, e.g. edit_d_refine_clamp
set(handles.edit_d_refine_clamp, 'String',num2str(funParams.diffusion.refine_clamp))
set(handles.edit_d_refine_iters, 'String',num2str(funParams.diffusion.refine_iters))
set(handles.edit_d_refine_alpha, 'String',num2str(funParams.diffusion.refine_alpha))
set(handles.edit_d_n_cpu, 'String',num2str(funParams.diffusion.n_cpu))
set(handles.edit_d_pad_size, 'String',num2str(funParams.diffusion.pad_size))
if funParams.diffusion.multilabel_refine
    set(handles.checkbox_d_multilabel_refine, 'Value', 1)
else
    set(handles.checkbox_d_multilabel_refine, 'Value', 0)
end

% "Guide Image" panel. tag names pattern as type_gi_parameterName, e.g. edit_gi_pmin
set(handles.edit_gi_pmin, 'String',num2str(funParams.guide_img.pmin))
set(handles.edit_gi_pmax, 'String',num2str(funParams.guide_img.pmax))

% "Ridge Filter" panel. tag names pattern as type_rf_parameterName, e.g. edit_rf_sigmas
set(handles.edit_rf_sigmas, 'String', num2str([funParams.ridge_filter.sigmas{:}])) % this is how to convert funParams.ridge_filter.sigmas = {3} to string.
set(handles.edit_rf_mix_ratio, 'String',num2str(funParams.ridge_filter.mix_ratio))
set(handles.edit_rf_low_contrast_fraction, 'String',num2str(funParams.ridge_filter.low_contrast_fraction))
set(handles.edit_rf_n_cpu, 'String',num2str(funParams.ridge_filter.n_cpu))
set(handles.edit_rf_pmin, 'String',num2str(funParams.ridge_filter.pmin))
set(handles.edit_rf_pmax, 'String',num2str(funParams.ridge_filter.pmax))
if funParams.ridge_filter.black_ridges
    set(handles.checkbox_rf_black_ridges, 'Value', 1)
else
    set(handles.checkbox_rf_black_ridges, 'Value', 0)
end
if funParams.ridge_filter.do_ridge_enhance
    set(handles.checkbox_rf_do_ridge_enhance, 'Value', 1)
else
    set(handles.checkbox_rf_do_ridge_enhance, 'Value', 0)
end
if funParams.ridge_filter.do_multiprocess_2D
    set(handles.checkbox_rf_do_multiprocess_2D, 'Value', 1)
else
    set(handles.checkbox_rf_do_multiprocess_2D, 'Value', 0)
end

% "Guide Filter" panel. tag names pattern as type_gf_parameterName, e.g. edit_gf_radius
set(handles.edit_gf_radius, 'String',num2str(funParams.guide_filter.radius))
set(handles.edit_gf_eps, 'String',num2str(funParams.guide_filter.eps))
set(handles.edit_gf_n_cpu, 'String',num2str(funParams.guide_filter.n_cpu))
set(handles.edit_gf_pad_size, 'String',num2str(funParams.guide_filter.pad_size))
set(handles.edit_gf_size_factor, 'String',num2str(funParams.guide_filter.size_factor))
set(handles.edit_gf_min_protrusion_size, 'String',num2str(funParams.guide_filter.min_protrusion_size))
if funParams.guide_filter.adaptive_radius_bool
    set(handles.checkbox_gf_adaptive_radius_bool, 'Value', 1)
else
    set(handles.checkbox_gf_adaptive_radius_bool, 'Value', 0)
end
%Setup "Operating Mode for Recovering Protrusions" pop up menu:
set(handles.popupmenu_gf_mode, 'String', SegmentationEnhancementPostprocessingProcess.getValidGuideFilterModename);
parVal = funParams.guide_filter.mode;
valSel  = find(ismember(SegmentationEnhancementPostprocessingProcess.getValidGuideFilterModename, parVal));
if isempty(valSel), valSel = 1; end
set(handles.popupmenu_gf_mode, 'Value', valSel);

set(handles.edit_gf_threshold_level, 'String',num2str(funParams.guide_filter.threshold_level))
set(handles.edit_gf_threshold_nlevels, 'String',num2str(funParams.guide_filter.threshold_nlevels))
if funParams.guide_filter.use_int
    set(handles.checkbox_gf_use_int, 'Value', 1)
else
    set(handles.checkbox_gf_use_int, 'Value', 0)
end
% "Collision" subpanel:
set(handles.edit_gf_collision_erode, 'String',num2str(funParams.guide_filter.collision_erode))
set(handles.edit_gf_collision_close, 'String',num2str(funParams.guide_filter.collision_close))
set(handles.edit_gf_collision_dilate, 'String',num2str(funParams.guide_filter.collision_dilate))
if funParams.guide_filter.collision_fill_holes
    set(handles.checkbox_gf_collision_fill_holes, 'Value', 1)
else
    set(handles.checkbox_gf_collision_fill_holes, 'Value', 0)
end
set(handles.edit_gf_base_dilate, 'String',num2str(funParams.guide_filter.base_dilate))
set(handles.edit_gf_base_erode, 'String',num2str(funParams.guide_filter.base_erode))
% "Guide Image" (sub)panel. tag names pattern as type_gi2_parameterName, e.g. edit_gi2_pmin
set(handles.edit_gi2_pmin, 'String',num2str(funParams.guide_img2.pmin))
set(handles.edit_gi2_pmax, 'String',num2str(funParams.guide_img2.pmax))


% Update user data and GUI data
handles.output = hObject;
set(handles.figure1, 'UserData', userData);
guidata(hObject, handles);





% --- Outputs from this function are returned to the command line.
function varargout = SegmentationEnhancementPostprocessingProcessGUI_OutputFcn(hObject, eventdata, handles)
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes during object deletion, before destroying properties.
function figure1_DeleteFcn(hObject, eventdata, handles)
userData = get(handles.figure1, 'UserData');
if isempty(userData), userData = struct(); end

if isfield(userData, 'helpFig') && ishandle(userData.helpFig)
   delete(userData.helpFig) 
end

set(handles.figure1, 'UserData', userData);
guidata(hObject,handles);


% --- Executes on key press with focus on figure1 and none of its controls.
function figure1_KeyPressFcn(hObject, eventdata, handles)
% hObject    handle to figure1 (see GCBO)
% eventdata  structure with the following fields (see MATLAB.UI.FIGURE)
%	Key: name of the key that was pressed, in lower case
%	Character: character interpretation of the key(s) that was pressed
%	Modifier: name(s) of the modifier key(s) (i.e., control, shift) pressed
% handles    structure with handles and user data (see GUIDATA)
if strcmp(eventdata.Key, 'return')
    pushbutton_done_Callback(handles.pushbutton_done, [], handles);
end


% --- Executes on button press in pushbutton_cancel.
function pushbutton_cancel_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_cancel (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
delete(handles.figure1);


% --- Executes on button press in pushbutton_done.
function pushbutton_done_Callback(hObject, eventdata, handles)

%  Check user input --------:
if isempty(get(handles.listbox_selectedChannels, 'String'))
   errordlg('Please select at least one input channel from ''Available Channels''.','Setting Error','modal') 
    return;
end

% "Diffusion" panel:
  % Clamping Ratio is float >=0 and <=1
if isnan(str2double(get(handles.edit_d_refine_clamp, 'String'))) ...
        || str2double(get(handles.edit_d_refine_clamp, 'String')) < 0 ...
        || str2double(get(handles.edit_d_refine_clamp, 'String')) > 1
    errordlg('Please provide a valid input for ''Clamping Ratio''.','Setting Error','modal');
    return;
end
  % Number of Iterations of Diffusion is interger >0
if isnan(str2double(get(handles.edit_d_refine_iters, 'String'))) ...
        || str2double(get(handles.edit_d_refine_iters, 'String')) <= 0 ...
        || floor(str2double(get(handles.edit_d_refine_iters, 'String'))) ~= str2double(get(handles.edit_d_refine_iters, 'String'))
    errordlg('Please provide a valid input for ''Number of Iterations of Diffusion''.','Setting Error','modal');
    return;
end
  % Weighting of Affinity Graph Between 0 and 1 is float >=0 and <=1
if isnan(str2double(get(handles.edit_d_refine_alpha, 'String'))) ...
        || str2double(get(handles.edit_d_refine_alpha, 'String')) < 0 ...
        || str2double(get(handles.edit_d_refine_alpha, 'String')) > 1
    errordlg('Please provide a valid input for ''Weighting of Affinity Graph Between 0 and 1''.','Setting Error','modal');
    return;
end
  % Number of CPU to Use (default is empty) can be empty (None, string(missing) in Matlab) or int >0
if ~isempty(get(handles.edit_d_n_cpu, 'String')) && (str2double(get(handles.edit_d_n_cpu, 'String')) <= 0 ...
        || floor(str2double(get(handles.edit_d_n_cpu, 'String'))) ~= str2double(get(handles.edit_d_n_cpu, 'String')))
    errordlg('Please provide a valid input for ''Number of CPU to Use in Diffusion panel''.','Setting Error','modal');
    return;
end
  % Number of Pixels to Pad is interger >=0
if isnan(str2double(get(handles.edit_d_pad_size, 'String'))) ...
        || str2double(get(handles.edit_d_pad_size, 'String')) < 0 ...
        || floor(str2double(get(handles.edit_d_pad_size, 'String'))) ~= str2double(get(handles.edit_d_pad_size, 'String'))
    errordlg('Please provide a valid input for ''Number of Pixels to Pad''.','Setting Error','modal');
    return;
end

% "Guide Image" panel
  % Lower Intensity Cutoffis float >=0 and <=100
if isnan(str2double(get(handles.edit_gi_pmin, 'String'))) ...
        || str2double(get(handles.edit_gi_pmin, 'String')) < 0 ...
        || str2double(get(handles.edit_gi_pmin, 'String')) > 100
    errordlg('Please provide a valid input for ''Lower Intensity Cutoff in Guide Image left panel''.','Setting Error','modal');
    return;
end
  % Upper Intensity Cutoffis float >=0 and <=100
if isnan(str2double(get(handles.edit_gi_pmax, 'String'))) ...
        || str2double(get(handles.edit_gi_pmax, 'String')) < 0 ...
        || str2double(get(handles.edit_gi_pmax, 'String')) > 100
    errordlg('Please provide a valid input for ''Upper Intensity Cutoff in Guide Image left panel''.','Setting Error','modal');
    return;
end

% "Ridge Filter" panel
  % List of Sizes [px] for Multiscale Ridge Filtering is a list of float >0 w/ at least one element
inputStr = get(handles.edit_rf_sigmas, 'String');
if isempty(inputStr) ...
    || any(isnan(sscanf(inputStr, '%f')')) ...
    || any(sscanf(inputStr, '%f')' <= 0) ...
    || length(sscanf(inputStr, '%d')') ~= length(strsplit(inputStr))
  errordlg('Please provide a valid input for ''List of Sizes [px] for Multiscale Ridge Filtering''.','Setting Error','modal');
  return;
end
  % Weighting of Ridge Filtered Guided Image is float >=0 and <=1
if isnan(str2double(get(handles.edit_rf_mix_ratio, 'String'))) ...
        || str2double(get(handles.edit_rf_mix_ratio, 'String')) < 0 ...
        || str2double(get(handles.edit_rf_mix_ratio, 'String')) > 1
    errordlg('Please provide a valid input for ''Weighting of Ridge Filtered Guided Image''.','Setting Error','modal');
    return;
end
  % Low-Contrast Image Cutoff is float >=0 and <=1
if isnan(str2double(get(handles.edit_rf_low_contrast_fraction, 'String'))) ...
        || str2double(get(handles.edit_rf_low_contrast_fraction, 'String')) < 0 ...
        || str2double(get(handles.edit_rf_low_contrast_fraction, 'String')) > 1
    errordlg('Please provide a valid input for ''Low-Contrast Image Cutoff''.','Setting Error','modal');
    return;
end
  % Number of CPU to Use (default is empty) can be empty (None, string(missing) in Matlab) or int >0
if ~isempty(get(handles.edit_rf_n_cpu, 'String')) && (str2double(get(handles.edit_rf_n_cpu, 'String')) <= 0 ...
        || floor(str2double(get(handles.edit_rf_n_cpu, 'String'))) ~= str2double(get(handles.edit_rf_n_cpu, 'String')))
    errordlg('Please provide a valid input for ''Number of CPU to Use in Ridge Filter panel''.','Setting Error','modal');
    return;
end
  % Lower Intensity Cutoffis float >=0 and <=100
if isnan(str2double(get(handles.edit_rf_pmin, 'String'))) ...
        || str2double(get(handles.edit_rf_pmin, 'String')) < 0 ...
        || str2double(get(handles.edit_rf_pmin, 'String')) > 100
    errordlg('Please provide a valid input for ''Lower Intensity Cutoff in Ridge Filter panel''.','Setting Error','modal');
    return;
end
  % Upper Intensity Cutoffis float >=0 and <=100
if isnan(str2double(get(handles.edit_rf_pmax, 'String'))) ...
        || str2double(get(handles.edit_rf_pmax, 'String')) < 0 ...
        || str2double(get(handles.edit_rf_pmax, 'String')) > 100
    errordlg('Please provide a valid input for ''Upper Intensity Cutoff in Ridge Filter panel''.','Setting Error','modal');
    return;
end

% "Guide Filter" panel
  % Radius is interger >0
if isnan(str2double(get(handles.edit_gf_radius, 'String'))) ...
        || str2double(get(handles.edit_gf_radius, 'String')) <= 0 ...
        || floor(str2double(get(handles.edit_gf_radius, 'String'))) ~= str2double(get(handles.edit_gf_radius, 'String'))
    errordlg('Please provide a valid input for ''Radius''.','Setting Error','modal');
    return;
end
  % Regularization Strength is float >0
if isnan(str2double(get(handles.edit_gf_eps, 'String'))) ...
        || str2double(get(handles.edit_gf_eps, 'String')) <= 0
    errordlg('Please provide a valid input for ''Regularization Strength''.','Setting Error','modal');
    return;
end
  % Number of CPU to Use (default is empty) can be empty (None, string(missing) in Matlab) or int >0
if ~isempty(get(handles.edit_gf_n_cpu, 'String')) && (str2double(get(handles.edit_gf_n_cpu, 'String')) <= 0 ...
        || floor(str2double(get(handles.edit_gf_n_cpu, 'String'))) ~= str2double(get(handles.edit_gf_n_cpu, 'String')))
    errordlg('Please provide a valid input for ''Number of CPU to Use in Guide Filter panel''.','Setting Error','modal');
    return;
end
  % Number of Pixels to Pad is interger >=0
if isnan(str2double(get(handles.edit_gf_pad_size, 'String'))) ...
        || str2double(get(handles.edit_gf_pad_size, 'String')) < 0 ...
        || floor(str2double(get(handles.edit_gf_pad_size, 'String'))) ~= str2double(get(handles.edit_gf_pad_size, 'String'))
    errordlg('Please provide a valid input for ''Number of Pixels to Pad''.','Setting Error','modal');
    return;
end
  % Proportion of Cell's Mean Bounding Box Diameter is float >0
if isnan(str2double(get(handles.edit_gf_size_factor, 'String'))) ...
        || str2double(get(handles.edit_gf_size_factor, 'String')) <= 0
    errordlg('Please provide a valid input for ''Proportion of Cells Mean Bounding Box Diameter'.','Setting Error','modal');
    return;
end
  % Minimum Protrusion Size [voxel] is float >=0
if isnan(str2double(get(handles.edit_gf_min_protrusion_size, 'String'))) ...
        || str2double(get(handles.edit_gf_min_protrusion_size, 'String')) < 0
    errordlg('Please provide a valid input for ''Minimum Protrusion Size [voxel]''.','Setting Error','modal');
    return;
end
  % Number of Binary Threshold Partitions is interger >=2
if isnan(str2double(get(handles.edit_gf_threshold_nlevels, 'String'))) ...
        || str2double(get(handles.edit_gf_threshold_nlevels, 'String')) < 2 ...
        || floor(str2double(get(handles.edit_gf_threshold_nlevels, 'String'))) ~= str2double(get(handles.edit_gf_threshold_nlevels, 'String'))
    errordlg('Please provide a valid input for ''Number of Binary Threshold Partitions''.','Setting Error','modal');
    return;
end
  % Binary Threshold Level is interger >=0, but <= "Number of Binary Threshold Partitions" - 2
if isnan(str2double(get(handles.edit_gf_threshold_level, 'String'))) ...
        || str2double(get(handles.edit_gf_threshold_level, 'String')) < 0 ...
        || floor(str2double(get(handles.edit_gf_threshold_level, 'String'))) ~= str2double(get(handles.edit_gf_threshold_level, 'String')) ...
        || str2double(get(handles.edit_gf_threshold_level, 'String')) > (str2double(get(handles.edit_gf_threshold_nlevels, 'String')) - 2)
    errordlg('Please provide a valid input for ''Binary Threshold Level''.','Setting Error','modal');
    return;
end
% "Collision" subpanel:
  % Protrusion Erosion Kernel Size is interger >=0
if isnan(str2double(get(handles.edit_gf_collision_erode, 'String'))) ...
        || str2double(get(handles.edit_gf_collision_erode, 'String')) < 0 ...
        || floor(str2double(get(handles.edit_gf_collision_erode, 'String'))) ~= str2double(get(handles.edit_gf_collision_erode, 'String'))
    errordlg('Please provide a valid input for ''Protrusion Erosion Kernel Size''.','Setting Error','modal');
    return;
end
  % Protrusion Closing Kernel Size is interger >=0
if isnan(str2double(get(handles.edit_gf_collision_close, 'String'))) ...
        || str2double(get(handles.edit_gf_collision_close, 'String')) < 0 ...
        || floor(str2double(get(handles.edit_gf_collision_close, 'String'))) ~= str2double(get(handles.edit_gf_collision_close, 'String'))
    errordlg('Please provide a valid input for ''Protrusion Closing Kernel Size''.','Setting Error','modal');
    return;
end
  % Protrusion Dilation Kernel Size is interger >=0
if isnan(str2double(get(handles.edit_gf_collision_dilate, 'String'))) ...
        || str2double(get(handles.edit_gf_collision_dilate, 'String')) < 0 ...
        || floor(str2double(get(handles.edit_gf_collision_dilate, 'String'))) ~= str2double(get(handles.edit_gf_collision_dilate, 'String'))
    errordlg('Please provide a valid input for ''Protrusion Dilation Kernel Size''.','Setting Error','modal');
    return;
end

  % Cell Dilation Kernel Size is interger >=0
if isnan(str2double(get(handles.edit_gf_base_dilate, 'String'))) ...
        || str2double(get(handles.edit_gf_base_dilate, 'String')) < 0 ...
        || floor(str2double(get(handles.edit_gf_base_dilate, 'String'))) ~= str2double(get(handles.edit_gf_base_dilate, 'String'))
    errordlg('Please provide a valid input for ''Cell Dilation Kernel Size''.','Setting Error','modal');
    return;
end
  % Cell Erosion Kernel Size is interger >=0
if isnan(str2double(get(handles.edit_gf_base_erode, 'String'))) ...
        || str2double(get(handles.edit_gf_base_erode, 'String')) < 0 ...
        || floor(str2double(get(handles.edit_gf_base_erode, 'String'))) ~= str2double(get(handles.edit_gf_base_erode, 'String'))
    errordlg('Please provide a valid input for ''Cell Erosion Kernel Size''.','Setting Error','modal');
    return;
end
% "Guide Image" (gi2) (sub)panel
  % Lower Intensity Cutoffis float >=0 and <=100
if isnan(str2double(get(handles.edit_gi2_pmin, 'String'))) ...
        || str2double(get(handles.edit_gi2_pmin, 'String')) < 0 ...
        || str2double(get(handles.edit_gi2_pmin, 'String')) > 100
    errordlg('Please provide a valid input for ''Lower Intensity Cutoff in Guide Image right panel''.','Setting Error','modal');
    return;
end
  % Upper Intensity Cutoffis float >=0 and <=100
if isnan(str2double(get(handles.edit_gi2_pmax, 'String'))) ...
        || str2double(get(handles.edit_gi2_pmax, 'String')) < 0 ...
        || str2double(get(handles.edit_gi2_pmax, 'String')) > 100
    errordlg('Please provide a valid input for ''Upper Intensity Cutoff in Guide Image right panel''.','Setting Error','modal');
    return;
end



%  Process Sanity check ( only check underlying data )
userData = get(handles.figure1, 'UserData');
if isempty(userData), userData = struct(); end
try
    userData.crtProc.sanityCheck;
catch ME
    errordlg([ME.message 'Please double check your data.'],...
                'Setting Error','modal');
    return;
end



% Retrieve GUI-defined parameters
channelIndex = get(handles.listbox_selectedChannels, 'Userdata');
funParams.ChannelIndex = channelIndex;

% "Diffusion" panel:
funParams.diffusion.refine_clamp = str2double(get(handles.edit_d_refine_clamp, 'String'));
funParams.diffusion.refine_iters = int16(str2double(get(handles.edit_d_refine_iters, 'String'))); % this parameter need to be a integer!
funParams.diffusion.refine_alpha = str2double(get(handles.edit_d_refine_alpha, 'String'));
  % this can be empty (None, string(missing) in Matlab) or integer:
if isempty(get(handles.edit_d_n_cpu, 'String'))
    funParams.diffusion.n_cpu = string(missing);
else
    funParams.diffusion.n_cpu = int16(str2double(get(handles.edit_d_n_cpu, 'String'))); % this parameter need to be a integer!
end
funParams.diffusion.pad_size = int16(str2double(get(handles.edit_d_pad_size, 'String'))); % this parameter need to be a integer!
if get(handles.checkbox_d_noprogress_bool, 'Value')
    funParams.diffusion.noprogress_bool = true;
else
    funParams.diffusion.noprogress_bool = false;
end
if get(handles.checkbox_d_multilabel_refine, 'Value')
    funParams.diffusion.multilabel_refine = true;
else
    funParams.diffusion.multilabel_refine = false;
end

% "Guide Image" panel
funParams.guide_img.pmin = str2double(get(handles.edit_gi_pmin, 'String'));
funParams.guide_img.pmax = str2double(get(handles.edit_gi_pmax, 'String'));

% "Ridge Filter" panel
funParams.ridge_filter.sigmas = num2cell(sscanf(get(handles.edit_rf_sigmas, 'String'), '%f')'); % need to be float ('%f') in 1xn (n>=1) cell array
funParams.ridge_filter.mix_ratio = str2double(get(handles.edit_rf_mix_ratio, 'String'));
funParams.ridge_filter.low_contrast_fraction = str2double(get(handles.edit_rf_low_contrast_fraction, 'String'));
  % this can be empty (None, string(missing) in Matlab) or integer:
if isempty(get(handles.edit_rf_n_cpu, 'String'))
    funParams.ridge_filter.n_cpu = string(missing);
else
    funParams.ridge_filter.n_cpu= int16(str2double(get(handles.edit_rf_n_cpu, 'String'))); % this parameter need to be a integer!
end
funParams.ridge_filter.pmin = str2double(get(handles.edit_rf_pmin, 'String'));
funParams.ridge_filter.pmax = str2double(get(handles.edit_rf_pmax, 'String'));
if get(handles.checkbox_rf_black_ridges, 'Value')
    funParams.ridge_filter.black_ridges = true;
else
    funParams.ridge_filter.black_ridges = false;
end
if get(handles.checkbox_rf_do_ridge_enhance, 'Value')
    funParams.ridge_filter.do_ridge_enhance = true;
else
    funParams.ridge_filter.do_ridge_enhance = false;
end
if get(handles.checkbox_rf_do_multiprocess_2D, 'Value')
    funParams.ridge_filter.do_multiprocess_2D = true;
else
    funParams.ridge_filter.do_multiprocess_2D = false;
end

% "Guide Filter" panel
funParams.guide_filter.radius = int16(str2double(get(handles.edit_gf_radius, 'String'))); % this parameter need to be a integer!
funParams.guide_filter.eps = str2double(get(handles.edit_gf_eps, 'String'));
  % this can be empty (None, string(missing) in Matlab) or integer:
if isempty(get(handles.edit_gf_n_cpu, 'String'))
    funParams.guide_filter.n_cpu = string(missing);
else
    funParams.guide_filter.n_cpu= int16(str2double(get(handles.edit_gf_n_cpu, 'String'))); % this parameter need to be a integer!
end
funParams.guide_filter.pad_size = int16(str2double(get(handles.edit_gf_pad_size, 'String'))); % this parameter need to be a integer!
funParams.guide_filter.size_factor = str2double(get(handles.edit_gf_size_factor, 'String'));
funParams.guide_filter.min_protrusion_size = str2double(get(handles.edit_gf_min_protrusion_size, 'String'));
if get(handles.checkbox_gf_adaptive_radius_bool, 'Value')
    funParams.guide_filter.adaptive_radius_bool = true;
else
    funParams.guide_filter.adaptive_radius_bool = false;
end

selType = get(handles.popupmenu_gf_mode, 'Value'); 
funParams.guide_filter.mode = SegmentationEnhancementPostprocessingProcess.getValidGuideFilterModename{selType};

funParams.guide_filter.threshold_level = int16(str2double(get(handles.edit_gf_threshold_level, 'String'))); % this parameter need to be a integer!
funParams.guide_filter.threshold_nlevels = int16(str2double(get(handles.edit_gf_threshold_nlevels, 'String'))); % this parameter need to be a integer!
if get(handles.checkbox_gf_use_int, 'Value')
    funParams.guide_filter.use_int = true;
else
    funParams.guide_filter.use_int = false;
end

% "Collision" subpanel:
funParams.guide_filter.collision_erode = int16(str2double(get(handles.edit_gf_collision_erode, 'String'))); % this parameter need to be a integer!
funParams.guide_filter.collision_close = int16(str2double(get(handles.edit_gf_collision_close, 'String'))); % this parameter need to be a integer!
funParams.guide_filter.collision_dilate = int16(str2double(get(handles.edit_gf_collision_dilate, 'String'))); % this parameter need to be a integer!
if get(handles.checkbox_gf_collision_fill_holes, 'Value')
    funParams.guide_filter.collision_fill_holes = true;
else
    funParams.guide_filter.collision_fill_holes = false;
end

funParams.guide_filter.base_dilate = int16(str2double(get(handles.edit_gf_base_dilate, 'String'))); % this parameter need to be a integer!
funParams.guide_filter.base_erode = int16(str2double(get(handles.edit_gf_base_erode, 'String'))); % this parameter need to be a integer!

% "Guide Image" (gi2) (sub)panel
funParams.guide_img2.pmin = str2double(get(handles.edit_gi2_pmin, 'String'));
funParams.guide_img2.pmax = str2double(get(handles.edit_gi2_pmax, 'String'));



% Set parameters and update main window
processGUI_ApplyFcn(hObject, eventdata, handles,funParams);




% --- Executes on key press with focus on pushbutton_done and none of its controls.
function pushbutton_done_KeyPressFcn(hObject, eventdata, handles)
% hObject    handle to pushbutton_done (see GCBO)
% eventdata  structure with the following fields (see MATLAB.UI.CONTROL.UICONTROL)
%	Key: name of the key that was pressed, in lower case
%	Character: character interpretation of the key(s) that was pressed
%	Modifier: name(s) of the modifier key(s) (i.e., control, shift) pressed
% handles    structure with handles and user data (see GUIDATA)
if strcmp(eventdata.Key, 'return')
    pushbutton_done_Callback(handles.pushbutton_done, [], handles);
end
