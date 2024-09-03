function varargout = thresholdProcessGUI(varargin)
% thresholdProcessGUI M-file for thresholdProcessGUI.fig
%      thresholdProcessGUI, by itself, creates a new thresholdProcessGUI or raises the existing
%      singleton*.
%
%      H = thresholdProcessGUI returns the handle to a new thresholdProcessGUI or the handle to
%      the existing singleton*.
%
%      thresholdProcessGUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in thresholdProcessGUI.M with the given input arguments.
%
%      thresholdProcessGUI('Property','Value',...) creates a new thresholdProcessGUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before thresholdProcessGUI_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to thresholdProcessGUI_OpeningFcn via varargin.
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

% Edit the above text to modify the response to help thresholdProcessGUI

% Last Modified by GUIDE v2.5 05-Sep-2017 17:50:52

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @thresholdProcessGUI_OpeningFcn, ...
                   'gui_OutputFcn',  @thresholdProcessGUI_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
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


% --- Executes just before thresholdProcessGUI is made visible.
function thresholdProcessGUI_OpeningFcn(hObject, eventdata, handles, varargin)

processGUI_OpeningFcn(hObject, eventdata, handles, varargin{:},'initChannel',0);

% ---------------------- Channel Setup -------------------------
userData = get(handles.figure1, 'UserData');
if isempty(userData), userData = struct(); end
funParams = userData.crtProc.funParams_;

% Set up available input channels
set(handles.listbox_availableChannels,'String',userData.MD.getChannelPaths(), ...
    'UserData',1:numel(userData.MD.channels_));

channelIndex = funParams.ChannelIndex;

% Find any parent process
userData.parentProc = userData.crtPackage.getParent(userData.procID);
if numel(userData.parentProc) > 1
    userData.parentProc = userData.parentProc(end); % use last parentProc as parentProc
end
if isempty(userData.crtPackage.processes_{userData.procID}) && ~isempty(userData.parentProc)
    % Check existence of all parent processes
    emptyParentProc = any(cellfun(@isempty,userData.crtPackage.processes_(userData.parentProc)));
    if ~emptyParentProc
        % Intersect channel index with channel index of parent processes
        parentChannelIndex = @(x) userData.crtPackage.processes_{x}.funParams_.ChannelIndex;
        for i = userData.parentProc
            channelIndex = intersect(channelIndex,parentChannelIndex(i));
        end
    end
   
end


if ~isempty(channelIndex)
    channelString = userData.MD.getChannelPaths(channelIndex);
else
    channelString = {};
end

set(handles.listbox_selectedChannels,'String',channelString,...
    'UserData',channelIndex);

set(handles.edit_GaussFilterSigma,'String',funParams.GaussFilterSigma);

threshMethods = userData.crtProc.getMethods();
set(handles.popupmenu_thresholdingMethod,'String',{threshMethods(:).name},...
    'Value',funParams.MethodIndx);

if(~isfield(funParams,'PreThreshold'))
    funParams.PreThreshold = false;
end

if(~isfield(funParams,'ExcludeOutliers'))
    funParams.ExcludeOutliers = false;
end

if(~isfield(funParams,'ExcludeZero'))
    funParams.ExcludeZero = false;
end

if(~isfield(funParams,'Invert'))
    funParams.Invert = false;
end

set(handles.checkbox_invert, 'Value', funParams.Invert);

useAutomatic = isempty(funParams.ThresholdValue) || funParams.PreThreshold;
useFixed = ~isempty(funParams.ThresholdValue) || funParams.PreThreshold;

if useAutomatic
   
    set(handles.checkbox_auto, 'Value', 1);
    set(get(handles.uipanel_automaticThresholding,'Children'),'Enable','on');
    if funParams.MaxJump
        set(handles.checkbox_max, 'Value', 1);
        set(handles.edit_jump, 'Enable','on','String',funParams.MaxJump);
    else
        set(handles.edit_jump, 'Enable', 'off');
    end
    if funParams.PreThreshold
        set(handles.checkbox_preThreshold, 'Value', 1);
    end
    
    if funParams.ExcludeOutliers
        set(handles.checkbox_exclude_outliers,'Value',1);
        set(handles.edit_exclude_outliers_k_sigma,'Enable','on','String',num2str(funParams.ExcludeOutliers));
    else
        set(handles.checkbox_exclude_outliers,'Value',0);
        set(handles.edit_exclude_outliers_k_sigma,'Enable','off','String','');
    end
    
    set(handles.checkbox_exclude_zero,'Value',funParams.ExcludeZero);      

else
    set(handles.checkbox_auto, 'Value', 0);
    set(get(handles.uipanel_automaticThresholding,'Children'),'Enable','off');
    
end

if useFixed    
    if(isempty(funParams.ThresholdValue))
        funParams.ThresholdValue(1) = 0;
    end
    if(isempty(funParams.IsPercentile))
        funParams.IsPercentile = false(size(funParams.ThresholdValue));
    end
%     set(handles.checkbox_auto, 'Value', 0);
    set(get(handles.uipanel_fixedThreshold,'Children'),'Enable','on');
%     set(get(handles.uipanel_automaticThresholding,'Children'),'Enable','off');
    set(handles.listbox_thresholdValues, 'String', thresholdsToString(funParams.ThresholdValue,funParams.IsPercentile));
    userData.thresholdValue=funParams.ThresholdValue(1);
    set(handles.slider_threshold, 'Value',funParams.ThresholdValue(1))
    set(handles.checkbox_is_percentile,'Value',funParams.IsPercentile(1));
else
    set(get(handles.uipanel_fixedThreshold,'Children'),'Enable','off');
    nSelectedChannels  = numel(get(handles.listbox_selectedChannels, 'String'));
    set(handles.listbox_thresholdValues, 'String',...
        num2cell(zeros(1,nSelectedChannels)));
    userData.thresholdValue=0;
end

% Initialize the frame number slider and eidt
nFrames=userData.MD.nFrames_;
if nFrames > 1
    set(handles.slider_frameNumber,'Value',1,'Min',1,...
        'Max',nFrames,'SliderStep',[1/double(nFrames)  10/double(nFrames)]);
else
    set(handles.slider_frameNumber,'Enable','off');
end
set(handles.text_nFrames,'String',['/ ' num2str(nFrames)]);
set(handles.edit_frameNumber,'Value',1);


% Initialize previewing constants
userData.previewFig =-1;
userData.chanIndx = 0;
userData.imIndx=0;

% Update user data and GUI data
handles.output = hObject;
set(hObject, 'UserData', userData);
guidata(hObject, handles);
update_data(hObject,eventdata,handles);


% --- Outputs from this function are returned to the command line.
function varargout = thresholdProcessGUI_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in pushbutton_cancel.
function pushbutton_cancel_Callback(hObject, eventdata, handles)
% Delete figure
delete(handles.figure1);


% --- Executes on button press in pushbutton_done.
function pushbutton_done_Callback(hObject, eventdata, handles)
% Call back function of 'Apply' button
userData = get(handles.figure1, 'UserData');
if isempty(userData), userData = struct(); end

% -------- Check user input --------
if isempty(get(handles.listbox_selectedChannels, 'String'))
   errordlg('Please select at least one input channel from ''Available Channels''.','Setting Error','modal') 
    return;
end
channelIndex = get (handles.listbox_selectedChannels, 'Userdata');
funParams.ChannelIndex = channelIndex;

gaussFilterSigma = str2double(get(handles.edit_GaussFilterSigma, 'String'));
if isnan(gaussFilterSigma) || gaussFilterSigma < 0
    errordlg(['Please provide a valid input for '''...
        get(handles.text_GaussFilterSigma,'String') '''.'],'Setting Error','modal');
    return;
end
funParams.GaussFilterSigma=gaussFilterSigma;

funParams.Invert = get(handles.checkbox_invert,'Value');

% Automatic and fixed are not mutually exclusive.
% If PreThreshold, one could use both automatic and fixed thresholds.
useAutomatic = get(handles.checkbox_auto, 'value');
useFixed = ~useAutomatic || get(handles.checkbox_preThreshold, 'value');

if useAutomatic
%     funParams.ThresholdValue = [ ];
    funParams.MethodIndx=get(handles.popupmenu_thresholdingMethod,'Value');
    
    if get(handles.checkbox_max, 'Value')
        % If both checkbox are checked
        maxJump=str2double(get(handles.edit_jump, 'String'));
        if isnan(maxJump) || maxJump < 1
            errordlg('Please provide a valid input for ''Maximum threshold jump''.','Setting Error','modal');
            return;
        end    
        funParams.MaxJump = str2double(get(handles.edit_jump,'String'));
    else
        funParams.MaxJump = 0;
    end
    
    if get(handles.checkbox_exclude_outliers, 'Value')
        excludeOutliersKSigma = str2double(get(handles.edit_exclude_outliers_k_sigma, 'String'));
        if isnan(excludeOutliersKSigma) || excludeOutliersKSigma <= 0
            errordlg('Please provide a valid input for ''Exclude Outliers Sigma''.','Setting Error','modal');
            return;
        end
        funParams.ExcludeOutliers = excludeOutliersKSigma;
    else
        funParams.ExcludeOutliers = 0;
    end
    
    if get(handles.checkbox_exclude_zero, 'Value')
        funParams.ExcludeZero = true;
    else
        funParams.ExcludeZero = false;
    end
    
    if useFixed
        % Use the fixed thresholds below before automatic thresholding
        funParams.PreThreshold = true;
    else
        % Threshold value will be determined only automatically
        funParams.PreThreshold = false;
        funParams.ThresholdValue = [ ];
    end
else
    % No automatic thresholding
    funParams.PreThreshold = false;
    % Since ~useAutomatic, useFixed is true and thus will be set below
    funParams.ThresholdValue = [ ];
    funParams.MaxJump = 0;
end

if useFixed
    threshold = get(handles.listbox_thresholdValues, 'String');
    if isempty(threshold)
       errordlg('Please provide at least one threshold value.','Setting Error','modal')
       return
    elseif length(threshold) ~= 1 && length(threshold) ~= length(channelIndex)
       errordlg('Please provide the same number of threshold values as the input channels.','Setting Error','modal')
       return
    else
        [threshold,isPercentile] = stringToThresholds(threshold);
%         threshold = str2double(threshold);
        if any(isnan(threshold))
            errordlg('Please provide valid threshold values.','Setting Error','modal')
            return            
        end
    end
    funParams.ThresholdValue = threshold;
    funParams.IsPercentile = isPercentile;
end
   

% -------- Process Sanity check --------
% ( only check underlying data )

try
    userData.crtProc.sanityCheck;
catch ME

    errordlg([ME.message 'Please double check your data.'],...
                'Setting Error','modal');
    return;
end

processGUI_ApplyFcn(hObject, eventdata, handles,funParams);

% --- Executes on button press in checkbox_all.
function checkbox_all_Callback(hObject, eventdata, handles)

% Hint: get(hObject,'Value') returns toggle state of checkbox_all
contents1 = get(handles.listbox_availableChannels, 'String');

chanIndex1 = get(handles.listbox_availableChannels, 'Userdata');
chanIndex2 = get(handles.listbox_selectedChannels, 'Userdata');

% Return if listbox1 is empty
if isempty(contents1)
    return;
end

switch get(hObject,'Value')
    case 1
        set(handles.listbox_selectedChannels, 'String', contents1);
        chanIndex2 = chanIndex1;
        thresholdValues =zeros(1,numel(chanIndex1));
    case 0
        set(handles.listbox_selectedChannels, 'String', {}, 'Value',1);
        chanIndex2 = [ ];
        thresholdValues = [];
end
set(handles.listbox_selectedChannels, 'UserData', chanIndex2);
set(handles.listbox_thresholdValues,'String',num2cell(thresholdValues))
update_data(hObject,eventdata,handles);

% --- Executes on button press in pushbutton_select.
function pushbutton_select_Callback(hObject, eventdata, handles)
% call back function of 'select' button

contents1 = get(handles.listbox_availableChannels, 'String');
contents2 = get(handles.listbox_selectedChannels, 'String');
id = get(handles.listbox_availableChannels, 'Value');

% If channel has already been added, return;
chanIndex1 = get(handles.listbox_availableChannels, 'Userdata');
chanIndex2 = get(handles.listbox_selectedChannels, 'Userdata');
[thresholdValues,isPercentile] = stringToThresholds(get(handles.listbox_thresholdValues,'String'));
% thresholdValues = cellfun(@str2num,get(handles.listbox_thresholdValues,'String'));

for i = id
    if any(strcmp(contents1{i}, contents2) )
        continue;
    else
        contents2{end+1} = contents1{i};
        thresholdValues(end+1) = 0;
        isPercentile(end+1) = false;
        chanIndex2 = cat(2, chanIndex2, chanIndex1(i));

    end
end

set(handles.listbox_selectedChannels, 'String', contents2, 'Userdata', chanIndex2);
set(handles.listbox_thresholdValues,'String',thresholdsToString(thresholdValues,isPercentile))
update_data(hObject,eventdata,handles);


% --- Executes on button press in pushbutton_delete.
function pushbutton_delete_Callback(hObject, eventdata, handles)
% Call back function of 'delete' button
contents = get(handles.listbox_selectedChannels,'String');
id = get(handles.listbox_selectedChannels,'Value');

% Return if list is empty
if isempty(contents) || isempty(id)
    return;
end

% Delete selected item
contents(id) = [ ];

% Delete userdata
chanIndex2 = get(handles.listbox_selectedChannels, 'Userdata');
chanIndex2(id) = [ ];
set(handles.listbox_selectedChannels, 'Userdata', chanIndex2);

% Update thresholdValues
% thresholdValues = cellfun(@str2num,get(handles.listbox_thresholdValues,'String'));
[thresholdValues,isPercentile] = stringToThresholds(get(handles.listbox_thresholdValues,'String'));
thresholdValues(id) = [];
isPercentile(id) = [];
set(handles.listbox_thresholdValues,'String',thresholdsToString(thresholdValues,isPercentile));

% Point 'Value' to the second last item in the list once the 
% last item has been deleted
if (id >length(contents) && id>1)
    set(handles.listbox_selectedChannels,'Value',length(contents));
    set(handles.listbox_thresholdValues,'Value',length(contents));
end
% Refresh listbox
set(handles.listbox_selectedChannels,'String',contents);
update_data(hObject,eventdata,handles);

% --- Executes on button press in checkbox_auto.
function checkbox_auto_Callback(hObject, eventdata, handles)
% Hint: get(hObject,'Value') returns toggle state of checkbox_auto
if get(hObject, 'Value')
    set(get(handles.uipanel_automaticThresholding,'Children'),'Enable','on');
    if ~get(handles.checkbox_preThreshold,'Value')
%         set(handles.edit_preThreshold,'Enable','off'); 
        set(get(handles.uipanel_fixedThreshold,'Children'),'Enable','off');
    end
    if ~get(handles.checkbox_max,'Value'), set(handles.edit_jump,'Enable','off'); end
else 
    set(get(handles.uipanel_automaticThresholding,'Children'),'Enable','off');
    set(get(handles.uipanel_fixedThreshold,'Children'),'Enable','on')
    set(handles.checkbox_max, 'Value', 0);
    set(handles.checkbox_preThreshold, 'Value', 0);
    set(handles.checkbox_applytoall, 'Value',0);
    set(handles.checkbox_preview, 'Value',1);
end
update_data(hObject,eventdata,handles);

% --- Executes during object deletion, before destroying properties.
function figure1_DeleteFcn(hObject, eventdata, handles)
% Notify the package GUI that the setting panel is closed
userData = get(handles.figure1, 'UserData');
if isempty(userData), userData = struct(); end

if ishandle(userData.helpFig), delete(userData.helpFig); end
if ishandle(userData.previewFig), delete(userData.previewFig); end

set(handles.figure1, 'UserData', userData);
guidata(hObject,handles);

% --- Executes on button press in checkbox_max.
function checkbox_max_Callback(hObject, eventdata, handles)

if get(hObject, 'value')
    set(handles.edit_jump, 'Enable', 'on');
else 
    set(handles.edit_jump, 'Enable', 'off');
end

function edit_preThreshold_Callback(hObject, eventdata, handles)
% hObject    handle to edit_preThreshold (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit_preThreshold as text
%        str2double(get(hObject,'String')) returns contents of edit_preThreshold as a double


% --- Executes during object creation, after setting all properties.
function edit_preThreshold_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit_preThreshold (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in checkbox_preThreshold.
function checkbox_preThreshold_Callback(hObject, eventdata, handles)
% hObject    handle to checkbox_preThreshold (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of checkbox_preThreshold
if get(hObject, 'value')
    set(get(handles.uipanel_fixedThreshold,'Children'),'Enable','on')
%     set(handles.checkbox_max, 'Value', 0);
%     set(handles.checkbox_preThreshold, 'Value', 0);
%     set(handles.checkbox_applytoall, 'Value',0);
    set(handles.checkbox_preview, 'Value',1);
else 
    set(get(handles.uipanel_fixedThreshold,'Children'),'Enable','off');
%     set(handles.edit_preThreshold, 'Enable', 'off');
end
update_data(hObject,eventdata,handles);


% --- Executes on key press with focus on pushbutton_done and none of its controls.
function pushbutton_done_KeyPressFcn(hObject, eventdata, handles)

if strcmp(eventdata.Key, 'return')
    pushbutton_done_Callback(handles.pushbutton_done, [], handles);
end

% --- Executes on key press with focus on figure1 and none of its controls.
function figure1_KeyPressFcn(hObject, eventdata, handles)

if strcmp(eventdata.Key, 'return')
    pushbutton_done_Callback(handles.pushbutton_done, [], handles);
end


% --- Executes on button press in pushbutton_set_threshold.
function pushbutton_set_threshold_Callback(hObject, eventdata, handles)

newThresholdValue = get(handles.slider_threshold,'Value');
newIsPercentile = get(handles.checkbox_is_percentile,'Value');
indx = get(handles.listbox_selectedChannels,'Value');
% thresholdValues = cellfun(@str2num,get(handles.listbox_thresholdValues,'String'));
[thresholdValues,isPercentile] = stringToThresholds(get(handles.listbox_thresholdValues,'String'));
thresholdValues(indx) = newThresholdValue;
isPercentile(indx) = newIsPercentile;
set(handles.listbox_thresholdValues,'String',thresholdsToString(thresholdValues,isPercentile));


% --- Executes on button press in checkbox_preview.
function checkbox_preview_Callback(hObject, eventdata, handles)

if get(handles.checkbox_preview,'Value'), 
    update_data(hObject,eventdata,handles); 
else
    userData = get(handles.figure1, 'UserData');
    if ishandle(userData.previewFig), delete(userData.previewFig); end
    % Save data and update graphics
    set(handles.figure1,'UserData',userData);
    guidata(hObject, handles);
end

function threshold_edition(hObject,eventdata, handles)

userData = get(handles.figure1, 'UserData');
if isempty(userData), userData = struct(); end

% Retrieve the valuye of the selected threshold
if strcmp(get(hObject,'Tag'),'edit_threshold')
    thresholdValue = str2double(get(handles.edit_threshold, 'String'));
else
    thresholdValue = get(handles.slider_threshold, 'Value');
end
thresholdValue=round(thresholdValue);

% Check the validity of the supplied threshold
if isnan(thresholdValue) || thresholdValue < get(handles.slider_threshold,'Min') || thresholdValue > get(handles.slider_threshold,'Max')
    warndlg('Please provide a valid coefficient.','Setting Error','modal');
    thresholdValue=userData.thresholdValue;
end

set(handles.slider_threshold,'Value',thresholdValue);
set(handles.edit_threshold,'String',thresholdValue);
    
% Save data and update graphics
set(handles.figure1,'UserData',userData);
guidata(hObject, handles);
update_data(hObject,eventdata,handles);


function imageNumber_edition(hObject,eventdata, handles)

% Retrieve the value of the selected image
if strcmp(get(hObject,'Tag'),'edit_imageNumber')
    imageNumber = str2double(get(handles.edit_frameNumber, 'String'));
else
    imageNumber = get(handles.slider_frameNumber, 'Value');
end
imageNumber=round(imageNumber);

% Check the validity of the supplied threshold
if isnan(imageNumber) || imageNumber < 0 || imageNumber > get(handles.slider_frameNumber,'Max')
    warndlg('Please provide a valid coefficient.','Setting Error','modal');
end

set(handles.slider_frameNumber,'Value',imageNumber);
set(handles.edit_frameNumber,'String',imageNumber);

% Save data and update graphics
guidata(hObject, handles);
update_data(hObject,eventdata,handles);


function update_data(hObject,eventdata, handles)

userData = get(handles.figure1, 'UserData');
if isempty(userData), userData = struct(); end

if strcmp(get(get(hObject,'Parent'),'Tag'),'uipanel_channels') ||...
        strcmp(get(hObject,'Tag'),'listbox_thresholdValues')
    % Check if changes have been at the list box level
    linkedListBoxes = {'listbox_selectedChannels','listbox_thresholdValues'};
    checkLinkBox = strcmp(get(hObject,'Tag'),linkedListBoxes);
    if any(checkLinkBox)
        value = get(handles.(linkedListBoxes{checkLinkBox}),'Value');
        set(handles.(linkedListBoxes{~checkLinkBox}),'Value',value);
    else
        value = get(handles.listbox_selectedChannels,'Value');
    end
    thresholdString = get(handles.listbox_thresholdValues,'String');
    if ~isempty(thresholdString)
%         thresholdValue = str2double(thresholdString{value});
        [thresholdValue,isPercentile] = stringToThresholds(thresholdString{value});
    else
        thresholdValue=0;
    end
    set(handles.edit_threshold,'String',thresholdValue);
    set(handles.slider_threshold,'Value',thresholdValue);
    setPercentileMode(handles,isPercentile);
    if isempty(thresholdString),
        if isfield(userData, 'previewFig'), delete(userData.previewFig); end
        set(handles.figure1, 'UserData', userData);
        guidata(hObject,handles);
        return
    end
end


% Retrieve the channex index
props=get(handles.listbox_selectedChannels,{'UserData','Value'});
if isempty(props{1}), return; end
chanIndx = props{1}(props{2});
imIndx = get(handles.slider_frameNumber,'Value');
thresholdValue = get(handles.slider_threshold, 'Value');

% Load a new image in case the image number or channel has been changed
if (chanIndx~=userData.chanIndx) ||  (imIndx~=userData.imIndx)
    if ~isempty(userData.parentProc) && ~isempty(userData.crtPackage.processes_{userData.parentProc}) &&...
            userData.crtPackage.processes_{userData.parentProc}.checkChannelOutput(chanIndx)
        userData.imData=userData.crtPackage.processes_{userData.parentProc}.loadOutImage(chanIndx,imIndx);
    else
        userData.imData=userData.MD.channels_(chanIndx).loadImage(imIndx);
    end
    
    % Get the value of the new maximum threshold
    if(get(handles.checkbox_is_percentile,'Value'))
        maxThresholdValue = 100;
        minThresholdValue = 0;
    else
        maxThresholdValue=max(max(userData.imData(:)),1);
        minThresholdValue=min(min(userData.imData(:)),0);
    end
    % Update the threshold Value if above the new maximum
    thresholdValue=min(thresholdValue,maxThresholdValue);
    
    set(handles.slider_threshold,'Value',thresholdValue,'Max',maxThresholdValue,...
        'Min',minThresholdValue, ...
        'SliderStep',[1/double(maxThresholdValue)  10/double(maxThresholdValue)]);
    set(handles.edit_threshold,'String',thresholdValue);
    
    userData.updateImage=1;
    userData.chanIndx=chanIndx;
    userData.imIndx=imIndx;
else
    userData.updateImage=0;
end


% Save the data
set(handles.figure1, 'UserData', userData);
guidata(hObject,handles);

% Update graphics if applicable
if get(handles.checkbox_preview,'Value')

    % Create figure if non-existing or closed
    if ~ishandle(userData.previewFig)
        userData.previewFig = figure('NumberTitle','off','Name','Thresholding preview',...
            'Position',[50 50 userData.MD.imSize_(2) userData.MD.imSize_(1)]);
        axes('Position',[0 0 1 1]);
    else
        figure(userData.previewFig);
    end
    
    % Retrieve the handle of the figures image
    imHandle = findobj(userData.previewFig,'Type','image');
    if userData.updateImage || isempty(imHandle)
        if isempty(imHandle)
            imHandle=imagesc(userData.imData);
            axis off;
        else
            set(imHandle,'CData',userData.imData);
        end
    end
    
    % Preview the tresholding output using the alphaData mapping    
    alphamask=ones(size(userData.imData));
    gaussFilterSigma = str2double(get(handles.edit_GaussFilterSigma, 'String'));
    if ~isnan(gaussFilterSigma) && gaussFilterSigma >0
        imData = filterGauss2D(userData.imData,gaussFilterSigma);
    else
        imData=userData.imData;
    end
    
    
    if get(handles.checkbox_auto,'Value')
        methodIndx=get(handles.popupmenu_thresholdingMethod,'Value');
        threshMethod = userData.crtProc.getMethods(methodIndx).func;
        
        try
            p.PreThreshold = get(handles.checkbox_preThreshold,'Value');
            p.ExcludeZero = get(handles.checkbox_exclude_zero,'Value');
            p.ExcludeOutliers = get(handles.checkbox_exclude_outliers,'Value') ...
                              * str2double(get(handles.edit_exclude_outliers_k_sigma,'String'));
            % iChan is not really 1, but we are reusing code from thresholdMovie              
            iChan = 1;
            p.ThresholdValue = get(handles.slider_threshold, 'Value');
            p.IsPercentile = get(handles.checkbox_is_percentile,'Value');
            p.Invert = get(handles.checkbox_invert,'Value');
            
            if(isnan(p.ExcludeOutliers))
                p.ExcludeOutliers = 0;
            end
            
            data = imData;
            
            %% Perform automatic thresholding
            if(p.PreThreshold)
                % Use automatic thresholding method only on pixels
                % above fixed threshold
                absoluteThreshold = p.ThresholdValue(iChan);
                if(p.IsPercentile(iChan))
                    absoluteThreshold = prctile(data(:),absoluteThreshold);
                end
                data = data(data > absoluteThreshold);
            end
            if(p.ExcludeZero)
                data = data(data ~= 0);
            end
            if(p.ExcludeOutliers)
                % Exclude outliers before perforing automatic
                % thresholding
                [~,inliers] = detectOutliers(data,p.ExcludeOutliers);
                data = data(inliers);
            end

            currThresh = threshMethod(data);

            if(p.PreThreshold)
                % Ensure absolute threshold is above fixed threshold
                currThresh = max(currThresh,absoluteThreshold);
            end
            if(p.ExcludeZero)
                % Threshold must be at least zero if excluding zeros
                currThresh = max(currThresh,0);
            end
            
            %% Use automatic thresholding

            if(p.Invert)
                alphamask(imData>=currThresh)=.4;
            else
                alphamask(imData<=currThresh)=.4;
            end
        catch err
            % To debug breakpoint here
            rethrow(err);
        end
    else
        % Preview manual threshold
        thresholdValue = get(handles.slider_threshold, 'Value');
        if(get(handles.checkbox_is_percentile,'Value'))
            thresholdValue = prctile(userData.imData(:),thresholdValue);
        end
        if(get(handles.checkbox_invert,'Value'))
            alphamask(imData>=thresholdValue)=.4;
        else
            alphamask(imData<=thresholdValue)=.4;
        end
    end
    set(imHandle,'AlphaData',alphamask,'AlphaDataMapping','none');


    
    set(handles.figure1, 'UserData', userData);
    guidata(hObject,handles);
end


% --- Executes on button press in checkbox_is_percentile.
function checkbox_is_percentile_Callback(hObject, eventdata, handles)
% hObject    handle to checkbox_is_percentile (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of checkbox_is_percentile
if get(handles.checkbox_is_percentile,'Value')
    % Convert threshold values into percentiles
    thresholdValue = get(handles.slider_threshold, 'Value');
    userData = get(handles.figure1, 'UserData');
    threshPercent = sum(userData.imData(:) < thresholdValue)/numel(userData.imData(:))*100;
    set(handles.slider_threshold,'Value',threshPercent,'Min',0,'Max',100, ...
        'SliderStep',[0.01 0.1] ...
        );
    set(handles.edit_threshold,'String',threshPercent);
    set(handles.text_percentile,'String','%');
    update_data(hObject,eventdata, handles)
else
    % Convert percentile to threshold values
    thresholdPercent = get(handles.slider_threshold, 'Value');
    userData = get(handles.figure1, 'UserData');
    thresholdValue = prctile(userData.imData(:),thresholdPercent);
    maxThresholdValue=max(max(userData.imData(:)),1);
    minThresholdValue=min(min(userData.imData(:)),0);
    set(handles.slider_threshold,'Value',thresholdValue,'Min',minThresholdValue,'Max',maxThresholdValue, ...
        'SliderStep',[1/double(maxThresholdValue)  10/double(maxThresholdValue)]);
    set(handles.edit_threshold,'String',thresholdValue);
    set(handles.text_percentile,'String','');
end

function strings = thresholdsToString(thresholdValues,isPercentile)
    strings = num2cell(thresholdValues);
    strings = cellfun(@num2str,strings,'UniformOutput',false);
    percent = cell(size(strings));
    [percent{isPercentile}] = deal('%');
    strings = strcat(strings,percent);
    
    
function [thresholdValues,isPercentile] = stringToThresholds(thresholdStrings)
    if(ischar(thresholdStrings))
        thresholdStrings = {thresholdStrings};
    end
    isPercentile = cellfun(@(x) x(end) == '%',thresholdStrings);
    thresholdStrings(isPercentile) = cellfun(@(x) x(1:end-1),thresholdStrings(isPercentile),'UniformOutput',false);
    thresholdValues = str2double(thresholdStrings);

function setPercentileMode(handles,isPercentile)
    percent = '%';
    set(handles.text_percentile,'String',percent(isPercentile));
    set(handles.checkbox_is_percentile,'Value',isPercentile);


% --- Executes on button press in checkbox_exclude_outliers.
function checkbox_exclude_outliers_Callback(hObject, eventdata, handles)
% hObject    handle to checkbox_exclude_outliers (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of checkbox_exclude_outliers
if get(hObject,'Value')
    set(handles.edit_exclude_outliers_k_sigma,'Enable','on','String','3');
else
    set(handles.edit_exclude_outliers_k_sigma,'Enable','off');
end


function edit_exclude_outliers_k_sigma_Callback(hObject, eventdata, handles)
% hObject    handle to edit_exclude_outliers_k_sigma (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit_exclude_outliers_k_sigma as text
%        str2double(get(hObject,'String')) returns contents of edit_exclude_outliers_k_sigma as a double
userData = get(hObject, 'UserData');
k_sigma = str2double(get(hObject,'String'));
try
    validateattributes(k_sigma,{'numeric'},{'nonnan','positive'})
    userData.k_sigma = k_sigma;
    set(hObject,'UserData',userData);
catch err
    warndlg('Please provide a valid sigma multiplier','Setting Error','modal');
    if(isempty(userData.k_sigma))
        userData.k_sigma = 3;
    end
    set(hObject,'String',userData.k_sigma);
end

% --- Executes during object creation, after setting all properties.
function edit_exclude_outliers_k_sigma_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit_exclude_outliers_k_sigma (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in checkbox_exclude_zero.
function checkbox_exclude_zero_Callback(hObject, eventdata, handles)
% hObject    handle to checkbox_exclude_zero (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of checkbox_exclude_zero


% --- Executes on button press in checkbox_invert.
function checkbox_invert_Callback(hObject, eventdata, handles)
% hObject    handle to checkbox_invert (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of checkbox_invert
update_data(hObject,eventdata,handles);
