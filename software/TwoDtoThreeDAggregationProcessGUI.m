function varargout = TwoDtoThreeDAggregationProcessGUI(varargin)
%TWODTOTHREEDAGGREGATIONPROCESSGUI MATLAB code file for TwoDtoThreeDAggregationProcessGUI.fig
%      TWODTOTHREEDAGGREGATIONPROCESSGUI, by itself, creates a new TWODTOTHREEDAGGREGATIONPROCESSGUI or raises the existing
%      singleton*.
%
%      H = TWODTOTHREEDAGGREGATIONPROCESSGUI returns the handle to a new TWODTOTHREEDAGGREGATIONPROCESSGUI or the handle to
%      the existing singleton*.
%
%      TWODTOTHREEDAGGREGATIONPROCESSGUI('Property','Value',...) creates a new TWODTOTHREEDAGGREGATIONPROCESSGUI using the
%      given property value pairs. Unrecognized properties are passed via
%      varargin to TwoDtoThreeDAggregationProcessGUI_OpeningFcn.  This calling syntax produces a
%      warning when there is an existing singleton*.
%
%      TWODTOTHREEDAGGREGATIONPROCESSGUI('CALLBACK') and TWODTOTHREEDAGGREGATIONPROCESSGUI('CALLBACK',hObject,...) call the
%      local function named CALLBACK in TWODTOTHREEDAGGREGATIONPROCESSGUI.M with the given input
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

% Edit the above text to modify the response to help TwoDtoThreeDAggregationProcessGUI

% Last Modified by GUIDE v2.5 24-Jul-2024 12:32:21

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @TwoDtoThreeDAggregationProcessGUI_OpeningFcn, ...
                   'gui_OutputFcn',  @TwoDtoThreeDAggregationProcessGUI_OutputFcn, ...
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


% --- Executes just before TwoDtoThreeDAggregationProcessGUI is made visible.
function TwoDtoThreeDAggregationProcessGUI_OpeningFcn(hObject, eventdata, handles, varargin)

processGUI_OpeningFcn(hObject, eventdata, handles, varargin{:},'initChannel',1);

% Parameter setup
userData = get(handles.figure1, 'UserData');
if isempty(userData), userData = struct(); end
funParams = userData.crtProc.funParams_;


% set GUI with Parameters:

% "Combine 2D to 3D Gradients" panel. tag names pattern as type_ccg_parameterName, e.g. edit_ccg_ksize
set(handles.edit_ccg_ksize, 'String',num2str(funParams.combine_cell_gradients.ksize))
set(handles.edit_ccg_alpha, 'String',num2str(funParams.combine_cell_gradients.alpha))
set(handles.edit_ccg_post_sigma, 'String',num2str(funParams.combine_cell_gradients.post_sigma))
set(handles.edit_ccg_smooth_sigma, 'String',num2str(funParams.combine_cell_gradients.smooth_sigma))

% "Combine 2D to 3D Probabilities" panel. tag names pattern as type_ccp_parameterName, e.g. edit_ccp_ksize
set(handles.edit_ccp_ksize, 'String',num2str(funParams.combine_cell_probs.ksize))
set(handles.edit_ccp_alpha, 'String',num2str(funParams.combine_cell_probs.alpha))
set(handles.edit_ccp_smooth_sigma, 'String',num2str(funParams.combine_cell_probs.smooth_sigma))
set(handles.edit_ccp_threshold_level, 'String',num2str(funParams.combine_cell_probs.threshold_level))
set(handles.edit_ccp_threshold_n_levels, 'String',num2str(funParams.combine_cell_probs.threshold_n_levels))
set(handles.edit_ccp_prob_thresh, 'String',num2str(funParams.combine_cell_probs.prob_thresh))
set(handles.edit_ccp_min_prob_thresh, 'String',num2str(funParams.combine_cell_probs.min_prob_thresh))
if funParams.combine_cell_probs.apply_one_d_p_thresh
    set(handles.checkbox_ccp_apply_one_d_p_thresh, 'Value', 1)
else
    set(handles.checkbox_ccp_apply_one_d_p_thresh, 'Value', 0)
end

% "Postprocess Combined 3D Probability" panel. tag names pattern as type_pb_parameterName, e.g. edit_pb_binary_closing
set(handles.edit_pb_binary_closing, 'String',num2str(funParams.postprocess_binary.binary_closing))
set(handles.edit_pb_remove_small_objects, 'String',num2str(funParams.postprocess_binary.remove_small_objects))
set(handles.edit_pb_binary_dilation, 'String',num2str(funParams.postprocess_binary.binary_dilation))
set(handles.edit_pb_extra_erode, 'String',num2str(funParams.postprocess_binary.extra_erode))
if funParams.postprocess_binary.binary_fill_holes
    set(handles.checkbox_pb_binary_fill_holes, 'Value', 1)
else
    set(handles.checkbox_pb_binary_fill_holes, 'Value', 0)
end

% "3D Gradient Descent" panel. tag names pattern as type_gd_parameterName, e.g. edit_gd_gradient_decay
set(handles.edit_gd_gradient_decay, 'String',num2str(funParams.gradient_descent.gradient_decay))
set(handles.edit_gd_ref_alpha, 'String',num2str(funParams.gradient_descent.ref_alpha))

    % set below 3 edit boxes anyway, even they depend on checkbox_gd_do_mp. b/c checkbox_gd_do_mp_Callback does not set them.
set(handles.edit_gd_tile_shape, 'String',num2str([funParams.gradient_descent.tile_shape{:}])) % this is how to convert funParams.gradient_descent.tile_shape = {int16(128), int16(256), int16(256)} to string.
set(handles.edit_gd_tile_aspect, 'String',num2str([funParams.gradient_descent.tile_aspect{:}])) % this is how to convert funParams.gradient_descent.tile_aspect = {1,2,2} to string.
set(handles.edit_gd_tile_overlap_ratio, 'String',num2str(funParams.gradient_descent.tile_overlap_ratio))
if funParams.gradient_descent.do_mp
    set(handles.checkbox_gd_do_mp, 'Value', 1);
else
    set(handles.checkbox_gd_do_mp, 'Value', 0);
    set(get(handles.uipanel_gd_do_mp,'Children'),'Enable','off');
end

% "3D Cell Clustering" panel. tag names pattern as type_cc_parameterName, e.g. edit_cc_min_area
set(handles.edit_cc_min_area, 'String',num2str(funParams.connected_component.min_area))
set(handles.edit_cc_smooth_sigma, 'String',num2str(funParams.connected_component.smooth_sigma))
set(handles.edit_cc_thresh_factor, 'String',num2str(funParams.connected_component.thresh_factor))

% "Distance Transform" panel. tag names pattern as type_im_parameterName, e.g. edit_im_power_dist
    %Setup Distance Transform Method pop up menu (popupmenu_im_dtform_method):
    % The 'String' of popupmenu_im_dtform_method does not match the funParams.indirect_method.dtform_method
    % So the 'String' of popupmenu_im_dtform_method was set in the GUI's property inspector.
parVal = funParams.indirect_method.dtform_method;
valSel  = find(ismember(TwoDtoThreeDAggregationProcess.getValidDistTransMethod, parVal));
if isempty(valSel), valSel = 1; end
set(handles.popupmenu_im_dtform_method, 'Value', valSel);

set(handles.edit_im_power_dist, 'String',num2str(funParams.indirect_method.power_dist))
set(handles.edit_im_edt_fixed_point_percentile, 'String',num2str(funParams.indirect_method.edt_fixed_point_percentile))
set(handles.edit_im_smooth_skel_sigma, 'String',num2str(funParams.indirect_method.smooth_skel_sigma))

set(handles.edit_im_n_cpu, 'String',num2str(funParams.indirect_method.n_cpu))
set(handles.edit_im_smooth_binary, 'String',num2str(funParams.indirect_method.smooth_binary))

    % Set up the dependency of this panel:
    % (1) gray out this panel if step 2 is not ExternalSegment3DProcess (indirect_method)
if isempty(funParams.ProcessIndex)
    iSegmentProc = userData.MD.getProcessIndex('ExternalSegment3DProcess', 'askUser', false, 'nDesired', Inf); % this can be multiple index
    if isempty(iSegmentProc)
        set(get(handles.uipanel_indirect_method,'Children'),'Enable','off');
    end
else
    if ~isa(userData.MD.processes_{funParams.ProcessIndex},'ExternalSegment3DProcess') % funParams.ProcessIndex is only one index
        set(get(handles.uipanel_indirect_method,'Children'),'Enable','off');
    end
end
    % (2) gray out Distance Transform Exponent if dtform_method is not 'cellpose_improve'
if ~isequal(funParams.indirect_method.dtform_method, 'cellpose_improve')
    set(handles.text_im_power_dist, 'Enable','off');
    set(handles.edit_im_power_dist, 'Enable','off');
end
    % (3) gray out Euclidean Distance Transform Threshold Percentile if dtform_method is not 'cellpose_improve' nor 'fmm'
if ~ismember(funParams.indirect_method.dtform_method, {'cellpose_improve', 'fmm'})
    set(handles.text_im_edt_fixed_point_percentile, 'Enable','off');
    set(handles.edit_im_edt_fixed_point_percentile, 'Enable','off');
end
    % (4) gray out Smooth Skeleton if dtform_method is not 'fmm_skel' nor 'cellpose_skel'
if ~ismember(funParams.indirect_method.dtform_method, {'fmm_skel', 'cellpose_skel'})
    set(handles.text_im_smooth_skel_sigma, 'Enable','off');
    set(handles.edit_im_smooth_skel_sigma, 'Enable','off');
end


% Update user data and GUI data
handles.output = hObject;
set(handles.figure1, 'UserData', userData);
guidata(hObject, handles);





% --- Outputs from this function are returned to the command line.
function varargout = TwoDtoThreeDAggregationProcessGUI_OutputFcn(hObject, eventdata, handles)
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

% "Combine 2D to 3D Gradients" panel:
  % Neighborhood Size is interger >=1
if isnan(str2double(get(handles.edit_ccg_ksize, 'String'))) ...
        || str2double(get(handles.edit_ccg_ksize, 'String')) < 1 ...
        || floor(str2double(get(handles.edit_ccg_ksize, 'String'))) ~= str2double(get(handles.edit_ccg_ksize, 'String'))
    errordlg('Please provide a valid input for ''Neighborhood Size''.','Setting Error','modal');
    return;
end

  % Pseudo Count Smoothing is float >0
if isnan(str2double(get(handles.edit_ccg_alpha, 'String'))) ...
        || str2double(get(handles.edit_ccg_alpha, 'String')) <= 0
    errordlg('Please provide a valid input for ''Pseudo Count Smoothing''.','Setting Error','modal');
    return;
end

  % Smooth Combined 3D Gradients is float >=0
if isnan(str2double(get(handles.edit_ccg_post_sigma, 'String'))) ...
        || str2double(get(handles.edit_ccg_post_sigma, 'String')) < 0
    errordlg('Please provide a valid input for ''Smooth Combined 3D Gradients''.','Setting Error','modal');
    return;
end

  % Presmooth 2D Gradients is float >=0
if isnan(str2double(get(handles.edit_ccg_smooth_sigma, 'String'))) ...
        || str2double(get(handles.edit_ccg_smooth_sigma, 'String')) < 0
    errordlg('Please provide a valid input for ''Presmooth 2D Gradients''.','Setting Error','modal');
    return;
end

% "Combine 2D to 3D Probabilities" panel:
  % Neighborhood Size is interger >=1
if isnan(str2double(get(handles.edit_ccp_ksize, 'String'))) ...
        || str2double(get(handles.edit_ccp_ksize, 'String')) < 1 ...
        || floor(str2double(get(handles.edit_ccp_ksize, 'String'))) ~= str2double(get(handles.edit_ccp_ksize, 'String'))
    errordlg('Please provide a valid input for ''Neighborhood Size''.','Setting Error','modal');
    return;
end

  % Pseudo Count Smoothing is float >0
if isnan(str2double(get(handles.edit_ccp_alpha, 'String'))) ...
        || str2double(get(handles.edit_ccp_alpha, 'String')) <= 0
    errordlg('Please provide a valid input for ''Pseudo Count Smoothing''.','Setting Error','modal');
    return;
end

  % Smooth Combined 3D Binary is float >=0
if isnan(str2double(get(handles.edit_ccp_smooth_sigma, 'String'))) ...
        || str2double(get(handles.edit_ccp_smooth_sigma, 'String')) < 0
    errordlg('Please provide a valid input for ''Smooth Combined 3D Binary''.','Setting Error','modal');
    return;
end

  % Number of Binary Threshold Partitions is interger >=2
if isnan(str2double(get(handles.edit_ccp_threshold_n_levels, 'String'))) ...
        || str2double(get(handles.edit_ccp_threshold_n_levels, 'String')) < 2 ...
        || floor(str2double(get(handles.edit_ccp_threshold_n_levels, 'String'))) ~= str2double(get(handles.edit_ccp_threshold_n_levels, 'String'))
    errordlg('Please provide a valid input for ''Number of Binary Threshold Partitions''.','Setting Error','modal');
    return;
end

  % Binary Threshold Level is interger >=0, but <= "Number of Binary Threshold Partitions" - 2
if isnan(str2double(get(handles.edit_ccp_threshold_level, 'String'))) ...
        || str2double(get(handles.edit_ccp_threshold_level, 'String')) < 0 ...
        || floor(str2double(get(handles.edit_ccp_threshold_level, 'String'))) ~= str2double(get(handles.edit_ccp_threshold_level, 'String')) ...
        || str2double(get(handles.edit_ccp_threshold_level, 'String')) > (str2double(get(handles.edit_ccp_threshold_n_levels, 'String')) - 2)
    errordlg('Please provide a valid input for ''Binary Threshold Level''.','Setting Error','modal');
    return;
end

  % Cell Probability Threshold can be empty (None, string(missing) in Matlab) or float >0 and <1
if ~isempty(get(handles.edit_ccp_prob_thresh, 'String')) && (str2double(get(handles.edit_ccp_prob_thresh, 'String')) <= 0 ...
        || str2double(get(handles.edit_ccp_prob_thresh, 'String')) >= 1)
    errordlg('Please provide a valid input for ''Cell Probability Threshold''.','Setting Error','modal');
    return;
end

  % Minimum Cell Probability Threshold is float >0 and <1
if isnan(str2double(get(handles.edit_ccp_min_prob_thresh, 'String'))) ...
        || str2double(get(handles.edit_ccp_min_prob_thresh, 'String')) < 0 ...
        || str2double(get(handles.edit_ccp_min_prob_thresh, 'String')) >= 1
    errordlg('Please provide a valid input for ''Minimum Cell Probability Threshold''.','Setting Error','modal');
    return;
end

% "Postprocess Combined 3D Probability" panel:
  % Binary Closing Kernel Size is interger >=0
if isnan(str2double(get(handles.edit_pb_binary_closing, 'String'))) ...
        || str2double(get(handles.edit_pb_binary_closing, 'String')) < 0 ...
        || floor(str2double(get(handles.edit_pb_binary_closing, 'String'))) ~= str2double(get(handles.edit_pb_binary_closing, 'String'))
    errordlg('Please provide a valid input for ''Binary Closing Kernel Size''.','Setting Error','modal');
    return;
end

  % Small Objects Size Cutoff [voxels] is interger >=0
if isnan(str2double(get(handles.edit_pb_remove_small_objects, 'String'))) ...
        || str2double(get(handles.edit_pb_remove_small_objects, 'String')) < 0 ...
        || floor(str2double(get(handles.edit_pb_remove_small_objects, 'String'))) ~= str2double(get(handles.edit_pb_remove_small_objects, 'String'))
    errordlg('Please provide a valid input for ''Small Objects Size Cutoff [voxels]''.','Setting Error','modal');
    return;
end

  % Binary Dilation Kernel Size is interger >=0
if isnan(str2double(get(handles.edit_pb_binary_dilation, 'String'))) ...
        || str2double(get(handles.edit_pb_binary_dilation, 'String')) < 0 ...
        || floor(str2double(get(handles.edit_pb_binary_dilation, 'String'))) ~= str2double(get(handles.edit_pb_binary_dilation, 'String'))
    errordlg('Please provide a valid input for ''Binary Dilation Kernel Size''.','Setting Error','modal');
    return;
end

  % Additional Binary Erosion Kernel Size is interger >=0
if isnan(str2double(get(handles.edit_pb_extra_erode, 'String'))) ...
        || str2double(get(handles.edit_pb_extra_erode, 'String')) < 0 ...
        || floor(str2double(get(handles.edit_pb_extra_erode, 'String'))) ~= str2double(get(handles.edit_pb_extra_erode, 'String'))
    errordlg('Please provide a valid input for ''Additional Binary Erosion Kernel Size''.','Setting Error','modal');
    return;
end

% "3D Gradient Descent" panel:
  % Temporal Decay is float >=0
if isnan(str2double(get(handles.edit_gd_gradient_decay, 'String'))) ...
        || str2double(get(handles.edit_gd_gradient_decay, 'String')) < 0
    errordlg('Please provide a valid input for ''Temporal Decay''.','Setting Error','modal');
    return;
end

  % Transparency of Visualized Points is float >=0 <=1
if isnan(str2double(get(handles.edit_gd_ref_alpha, 'String'))) ...
        || str2double(get(handles.edit_gd_ref_alpha, 'String')) < 0 ...
        || str2double(get(handles.edit_gd_ref_alpha, 'String')) > 1
    errordlg('Please provide a valid input for ''Transparency of Visualized Points''.','Setting Error','modal');
    return;
end

if get(handles.checkbox_gd_do_mp, 'value')
    % Subvolume Shape needs to be 3 positive integers
    inputStr = get(handles.edit_gd_tile_shape, 'String');
    if isempty(inputStr) ...
        || any(isnan(sscanf(inputStr, '%f')')) ...
        || any(sscanf(inputStr, '%f')' <= 0) ...
        || any(floor(sscanf(inputStr, '%f')') ~= sscanf(inputStr, '%f')') ...
        || length(sscanf(inputStr, '%f')') ~= 3 ...
        || length(sscanf(inputStr, '%f')') ~= length(strsplit(inputStr))
      errordlg('Please provide a valid input for ''Subvolume Shape''.','Setting Error','modal');
      return;
    end

    % Subvolume Aspect Ratio needs to be 3 positive floats
    inputStr = get(handles.edit_gd_tile_aspect, 'String');
    if isempty(inputStr) ...
        || any(isnan(sscanf(inputStr, '%f')')) ...
        || any(sscanf(inputStr, '%f')' <= 0) ...
        || length(sscanf(inputStr, '%f')') ~= 3 ...
        || length(sscanf(inputStr, '%f')') ~= length(strsplit(inputStr))    
      errordlg('Please provide a valid input for ''Subvolume Aspect Ratio''.','Setting Error','modal');
      return;
    end

    % Subvolume Overlap Fraction is float >=0 <1
    if isnan(str2double(get(handles.edit_gd_tile_overlap_ratio, 'String'))) ...
            || str2double(get(handles.edit_gd_tile_overlap_ratio, 'String')) < 0 ...
            || str2double(get(handles.edit_gd_tile_overlap_ratio, 'String')) >= 1
        errordlg('Please provide a valid input for ''Subvolume Overlap Fraction''.','Setting Error','modal');
        return;
    end 
end

% "3D Cell Clustering" panel:
  % Smallest Cluster Size Cutoff is interger >=0
if isnan(str2double(get(handles.edit_cc_min_area, 'String'))) ...
        || str2double(get(handles.edit_cc_min_area, 'String')) < 0 ...
        || floor(str2double(get(handles.edit_cc_min_area, 'String'))) ~= str2double(get(handles.edit_cc_min_area, 'String'))
    errordlg('Please provide a valid input for ''Smallest Cluster Size Cutoff''.','Setting Error','modal');
    return;
end

  % Smoothing Sigma for Computing Point Density is float >=0
if isnan(str2double(get(handles.edit_cc_smooth_sigma, 'String'))) ...
        || str2double(get(handles.edit_cc_smooth_sigma, 'String')) < 0
    errordlg('Please provide a valid input for ''Smoothing Sigma for Computing Point Density''.','Setting Error','modal');
    return;
end

  % Thresholding Adjustment Factor is float >=0
if isnan(str2double(get(handles.edit_cc_thresh_factor, 'String'))) ...
        || str2double(get(handles.edit_cc_thresh_factor, 'String')) < 0
    errordlg('Please provide a valid input for ''Thresholding Adjustment Factor''.','Setting Error','modal');
    return;
end


% "Distance Transform" panel:
% check only when the elements are not gray out:

if isequal(get(handles.edit_im_power_dist, 'Enable'), 'on')
    % Distance Transform Exponent can be empty (None, string(missing) in Matlab) or float >0 and <=1
  if ~isempty(get(handles.edit_im_power_dist, 'String')) && (str2double(get(handles.edit_im_power_dist, 'String')) <= 0 ...
          || str2double(get(handles.edit_im_power_dist, 'String')) > 1)
      errordlg('Please provide a valid input for ''Distance Transform Exponent''.','Setting Error','modal');
      return;
  end
end

if isequal(get(handles.edit_im_edt_fixed_point_percentile, 'Enable'), 'on')
    % Euclidean Distance Transform Threshold Percentile is float >=0 and <=1
  if isnan(str2double(get(handles.edit_im_edt_fixed_point_percentile, 'String'))) ...
          || str2double(get(handles.edit_im_edt_fixed_point_percentile, 'String')) < 0 ...
          || str2double(get(handles.edit_im_edt_fixed_point_percentile, 'String')) > 1
      errordlg('Please provide a valid input for ''Euclidean Distance Transform Threshold Percentile''.','Setting Error','modal');
      return;
  end
end

if isequal(get(handles.edit_im_smooth_skel_sigma, 'Enable'), 'on')
    % Smooth Skeleton is float >=1
  if isnan(str2double(get(handles.edit_im_smooth_skel_sigma, 'String'))) ...
          || str2double(get(handles.edit_im_smooth_skel_sigma, 'String')) < 1
      errordlg('Please provide a valid input for ''Smooth Skeleton''.','Setting Error','modal');
      return;
  end
end

if isequal(get(handles.edit_im_n_cpu, 'Enable'), 'on')
    % Number of CPU to Use (default is empty) can be empty (None, string(missing) in Matlab) or int >0
  if ~isempty(get(handles.edit_im_n_cpu, 'String')) && (str2double(get(handles.edit_im_n_cpu, 'String')) <= 0 ...
          || floor(str2double(get(handles.edit_im_n_cpu, 'String'))) ~= str2double(get(handles.edit_im_n_cpu, 'String')))
      errordlg('Please provide a valid input for ''Number of CPU to Use (default is empty)''.','Setting Error','modal');
      return;
  end
end

if isequal(get(handles.edit_im_smooth_binary, 'Enable'), 'on')
    % Smooth Combined 3D Binary is float >=1
  if isnan(str2double(get(handles.edit_im_smooth_binary, 'String'))) ...
          || str2double(get(handles.edit_im_smooth_binary, 'String')) < 1
      errordlg('Please provide a valid input for ''Smooth Combined 3D Binary''.','Setting Error','modal');
      return;
  end
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

% "Combine 2D to 3D Gradients" panel:
funParams.combine_cell_gradients.ksize = int16(str2double(get(handles.edit_ccg_ksize, 'String'))); % this parameter need to be a integer!
funParams.combine_cell_gradients.alpha = str2double(get(handles.edit_ccg_alpha, 'String'));
funParams.combine_cell_gradients.post_sigma = str2double(get(handles.edit_ccg_post_sigma, 'String'));
funParams.combine_cell_gradients.smooth_sigma = str2double(get(handles.edit_ccg_smooth_sigma, 'String'));

% "Combine 2D to 3D Probabilities" panel:
funParams.combine_cell_probs.ksize = int16(str2double(get(handles.edit_ccp_ksize, 'String'))); % this parameter need to be a integer!
funParams.combine_cell_probs.alpha = str2double(get(handles.edit_ccp_alpha, 'String'));
funParams.combine_cell_probs.smooth_sigma = str2double(get(handles.edit_ccp_smooth_sigma, 'String'));
funParams.combine_cell_probs.threshold_level = int16(str2double(get(handles.edit_ccp_threshold_level, 'String'))); % this parameter need to be a integer!
funParams.combine_cell_probs.threshold_n_levels = int16(str2double(get(handles.edit_ccp_threshold_n_levels, 'String'))); % this parameter need to be a integer!
if get(handles.checkbox_ccp_apply_one_d_p_thresh, 'Value')
    funParams.combine_cell_probs.apply_one_d_p_thresh = true;
else
    funParams.combine_cell_probs.apply_one_d_p_thresh = false;
end
  % funParams.combine_cell_probs.prob_thresh can be empty (None, string(missing) in Matlab) or float:
if isempty(get(handles.edit_ccp_prob_thresh, 'String'))
    funParams.combine_cell_probs.prob_thresh = string(missing);
else
    funParams.combine_cell_probs.prob_thresh = str2double(get(handles.edit_ccp_prob_thresh, 'String'));
end
funParams.combine_cell_probs.min_prob_thresh = str2double(get(handles.edit_ccp_min_prob_thresh, 'String'));

% "Postprocess Combined 3D Probability" panel:
funParams.postprocess_binary.binary_closing = int16(str2double(get(handles.edit_pb_binary_closing, 'String'))); % this parameter need to be a integer!
funParams.postprocess_binary.remove_small_objects = int16(str2double(get(handles.edit_pb_remove_small_objects, 'String'))); % this parameter need to be a integer!
funParams.postprocess_binary.binary_dilation = int16(str2double(get(handles.edit_pb_binary_dilation, 'String'))); % this parameter need to be a integer!
funParams.postprocess_binary.extra_erode = int16(str2double(get(handles.edit_pb_extra_erode, 'String'))); % this parameter need to be a integer!
if get(handles.checkbox_pb_binary_fill_holes, 'Value')
    funParams.postprocess_binary.binary_fill_holes = true;
else
    funParams.postprocess_binary.binary_fill_holes = false;
end

% "3D Gradient Descent" panel:
funParams.gradient_descent.gradient_decay = str2double(get(handles.edit_gd_gradient_decay, 'String'));
funParams.gradient_descent.ref_alpha = str2double(get(handles.edit_gd_ref_alpha, 'String'));
if get(handles.checkbox_gd_do_mp, 'Value')
    funParams.gradient_descent.do_mp = true;
    funParams.gradient_descent.tile_shape = num2cell(sscanf(get(handles.edit_gd_tile_shape, 'String'), '%d')'); % need to be 3 integer ('%d') in 1x3 cell array
    funParams.gradient_descent.tile_aspect = num2cell(sscanf(get(handles.edit_gd_tile_aspect, 'String'), '%f')'); % need to be 3 float ('%f') in 1x3 cell array
    funParams.gradient_descent.tile_overlap_ratio = str2double(get(handles.edit_gd_tile_overlap_ratio, 'String'));
else
    funParams.gradient_descent.do_mp = false;
end

% "3D Cell Clustering" panel:
funParams.connected_component.min_area = int16(str2double(get(handles.edit_cc_min_area, 'String'))); % this parameter need to be a integer!
funParams.connected_component.smooth_sigma = str2double(get(handles.edit_cc_smooth_sigma, 'String'));
funParams.connected_component.thresh_factor = str2double(get(handles.edit_cc_thresh_factor, 'String'));

% "Distance Transform" panel:
% Retrieve GUI-defined parameters in this panel when the elements are not gray out:
if isequal(get(handles.popupmenu_im_dtform_method, 'Enable'), 'on')
  selType = get(handles.popupmenu_im_dtform_method, 'Value'); 
  funParams.indirect_method.dtform_method = TwoDtoThreeDAggregationProcess.getValidDistTransMethod{selType};
end

if isequal(get(handles.edit_im_power_dist, 'Enable'), 'on')
    % this can be empty (None, string(missing) in Matlab) or float:
  if isempty(get(handles.edit_im_power_dist, 'String'))
      funParams.indirect_method.power_dist = string(missing);
  else
      funParams.indirect_method.power_dist = str2double(get(handles.edit_im_power_dist, 'String'));
  end
end

if isequal(get(handles.edit_im_edt_fixed_point_percentile, 'Enable'), 'on')
  funParams.indirect_method.edt_fixed_point_percentile = str2double(get(handles.edit_im_edt_fixed_point_percentile, 'String'));
end

if isequal(get(handles.edit_im_smooth_skel_sigma, 'Enable'), 'on')
  funParams.indirect_method.smooth_skel_sigma = str2double(get(handles.edit_im_smooth_skel_sigma, 'String'));
end

if isequal(get(handles.edit_im_n_cpu, 'Enable'), 'on')
    % this can be empty (None, string(missing) in Matlab) or integer:
  if isempty(get(handles.edit_im_n_cpu, 'String'))
      funParams.indirect_method.n_cpu = string(missing);
  else
      funParams.indirect_method.n_cpu = int16(str2double(get(handles.edit_im_n_cpu, 'String'))); % this parameter need to be a integer!
  end
end

if isequal(get(handles.edit_im_smooth_binary, 'Enable'), 'on')
  funParams.indirect_method.smooth_binary = str2double(get(handles.edit_im_smooth_binary, 'String'));
end



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


% --- Executes on button press in checkbox_gd_do_mp.
function checkbox_gd_do_mp_Callback(hObject, eventdata, handles)
% Hint: get(hObject,'Value') returns toggle state of checkbox_gd_do_mp
if get(hObject, 'Value')
    set(get(handles.uipanel_gd_do_mp,'Children'),'Enable','on');
else
    set(get(handles.uipanel_gd_do_mp,'Children'),'Enable','off');
end


% --- Executes on selection change in popupmenu_im_dtform_method.
function popupmenu_im_dtform_method_Callback(hObject, eventdata, handles)
% Hints: contents = cellstr(get(hObject,'String')) returns popupmenu_im_dtform_method contents as cell array
%        contents{get(hObject,'Value')} returns selected item from popupmenu_im_dtform_method

% Default all controls to 'off'
set([handles.text_im_power_dist, handles.edit_im_power_dist, ...
     handles.text_im_edt_fixed_point_percentile, handles.edit_im_edt_fixed_point_percentile, ...
     handles.text_im_smooth_skel_sigma, handles.edit_im_smooth_skel_sigma], 'Enable', 'off');

% set some controls to 'on' in below conditions
switch get(hObject, 'Value')
    case 1
        set([handles.text_im_power_dist, handles.edit_im_power_dist, ...
             handles.text_im_edt_fixed_point_percentile, handles.edit_im_edt_fixed_point_percentile], 'Enable', 'on');
    case 3
        set([handles.text_im_edt_fixed_point_percentile, handles.edit_im_edt_fixed_point_percentile], 'Enable', 'on');
    case {4, 5}
        set([handles.text_im_smooth_skel_sigma, handles.edit_im_smooth_skel_sigma], 'Enable', 'on');
end
