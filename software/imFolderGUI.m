function varargout = imFolderGUI(varargin)
%IMFOLDERGUI MATLAB code file for imFolderGUI.fig
%      IMFOLDERGUI, by itself, creates a new IMFOLDERGUI or raises the existing
%      singleton*.
%
%      H = IMFOLDERGUI returns the handle to a new IMFOLDERGUI or the handle to
%      the existing singleton*.
%
%      IMFOLDERGUI('Property','Value',...) creates a new IMFOLDERGUI using the
%      given property value pairs. Unrecognized properties are passed via
%      varargin to imFolderGUI_OpeningFcn.  This calling syntax produces a
%      warning when there is an existing singleton*.
%
%      IMFOLDERGUI('CALLBACK') and IMFOLDERGUI('CALLBACK',hObject,...) call the
%      local function named CALLBACK in IMFOLDERGUI.M with the given input
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

% Edit the above text to modify the response to help imFolderGUI

% Last Modified by GUIDE v2.5 07-Jul-2020 13:36:24

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @imFolderGUI_OpeningFcn, ...
                   'gui_OutputFcn',  @imFolderGUI_OutputFcn, ...
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


% --- Executes just before imFolderGUI is made visible.
function imFolderGUI_OpeningFcn(hObject, eventdata, handles, varargin)
%
% imFolderGUI('mainFig', handles.figure1) - call from imageDataGUI
% imFolderGUI(imFolderArray) - call from command line
% imFolderGUI(..., 'modal') - call imFolderGUI as a modal window
%
% User Data:
% 
% userData.imFolders - array of imFolder object
% userData.selectedImFolder - index of current imFolder
% userData.mainFig - handle of imageDataGUI
% userData.properties - structure array of properties 
% userData.propNames - list of modifiable properties
%
% userData.helpFig - handle of help window
%

set(handles.text_copyright, 'String', getLCCBCopyright())

userData = get(handles.figure1, 'UserData');

if isempty(userData), userData = struct(); end

% Choose default command line output for imFolderGUI
handles.output = hObject;

% Load help icon from dialogicons.mat
userData = loadLCCBIcons(userData);
supermap(1,:) = get(hObject,'color');

userData.colormap = supermap;

set(handles.figure1,'CurrentAxes',handles.axes_help);
Img = image(userData.questIconData);
set(hObject,'colormap',supermap);
set(gca, 'XLim',get(Img,'XData'),'YLim',get(Img,'YData'),...
    'visible','off');
set(Img,'ButtonDownFcn',@icon_ButtonDownFcn,...
    'UserData', struct('class', mfilename))

if nargin > 3    
    if any(strcmp(varargin, 'mainFig'))
        % Called from imageDataGUI
        
        % Get userData.mainFig
        t = find(strcmp(varargin, 'mainFig'));
        userData.mainFig = varargin{t(end)+1};
        
        % Get userData.imFolders
        userData_main = get(userData.mainFig, 'UserData');
        assert(isa(userData_main.imFolders(1), 'ImFolder'), 'User-defined: No ImFolder object found.')
        userData.imFolders = userData_main.imFolders;
        
        % Get userData.selectedImFolder
        handles_main = guidata(userData.mainFig);
        userData.selectedImFolder = get(handles_main.listbox_channel, 'Value');
        
    elseif isa(varargin{1}(1), 'ImFolder')
        % Called from command line        
        userData.imFolders = varargin{1};
        userData.selectedImFolder = 1;        
    else
        error('User-defined: Input parameters are incorrect.')
    end
    
    % Set as modal window
    if any(strcmp(varargin, 'modal'))
        set(hObject, 'WindowStyle', 'modal')
    end
    
else
    error('User-defined: No proper input.')
end

% Set up imaging mode and flurophore pop-up menu
modeString = vertcat({''},ImFolder.getImagingModes());
set(handles.popupmenu_imageType,'String',modeString);
% fluorophoreString = horzcat({''},ImFolder.getFluorophores());
% set(handles.popupmenu_fluorophore,'String',fluorophoreString);

% Read imFolder initial properties and store them in a structure
propNames={'pixelSize_','imageType_'};
userData.isNumProp = @(x) x<2;
for i=1:numel(userData.imFolders)
    for j=1:numel(propNames)
        userData.properties(i).(propNames{j})=...
            userData.imFolders(i).(propNames{j});
    end
end
% propNames={'excitationWavelength_','emissionWavelength_',...
%     'exposureTime_','imageType_','fluorophore_'};
% userData.isNumProp = @(x) x<4;
% for i=1:numel(userData.imFolders)
%     for j=1:numel(propNames)
%         userData.properties(i).(propNames{j})=...
%             userData.imFolders(i).(propNames{j});
%     end
% end

% Set up pop-up menu
set(handles.popupmenu_channel, 'String', ... 
arrayfun(@(x)(['ImFolder ' num2str(x)]), 1:length(userData.imFolders), 'UniformOutput', false) )
set(handles.popupmenu_channel, 'Value', userData.selectedImFolder)

% Set up imFolder path and properties
set(handles.figure1,'Visible','on');
set(handles.text_path, 'String', userData.imFolders(userData.selectedImFolder).imFolderPath_)
for i=1:numel(propNames)
    propValue = userData.properties(userData.selectedImFolder).(propNames{i});
    if userData.isNumProp(i)
        propHandle = handles.(['edit_' propNames{i}(1:end-1)]);
        guiProp = 'String';
        guiValue = propValue;
    else
        propHandle = handles.(['popupmenu_' propNames{i}(1:end-1)]);
        guiProp = 'Value';
        guiValue = find(strcmpi(propValue,get(propHandle,'String')));
        if isempty(guiValue), guiValue=1; end
    end
    if ~isempty(propValue), enableState='inactive'; else enableState='on'; end
    set(propHandle,guiProp,guiValue,'Enable',enableState);
end

% % Display psf sigma if computed
% if ~isempty(userData.imFolders(userData.selectedImFolder).psfSigma_),
%     set(handles.edit_psfSigma, 'String',...
%         userData.imFolders(userData.selectedImFolder).psfSigma_);
% else
%     set(handles.edit_psfSigma, 'String', '');
% end

% Update handles structure
set(handles.figure1,'UserData',userData)
guidata(hObject, handles);

% UIWAIT makes imFolderGUI wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = imFolderGUI_OutputFcn(hObject, eventdata, handles)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on key press with focus on figure1 and none of its controls.
function figure1_KeyPressFcn(hObject, eventdata, handles)
if strcmp(eventdata.Key, 'return')
    pushbutton_done_Callback(handles.pushbutton_done, [], handles);
end


% --- Executes on button press in pushbutton_cancel.
function pushbutton_cancel_Callback(hObject, eventdata, handles)

delete(handles.figure1)


% --- Executes on button press in pushbutton_done.
function pushbutton_done_Callback(hObject, eventdata, handles)

userData = get(handles.figure1, 'UserData');

% Retrieve names and handles of non-empty fields
propNames = fieldnames(userData.properties(1));
for i=1:numel(propNames)
    if userData.isNumProp(i)
        propHandle = handles.(['edit_' propNames{i}(1:end-1)]);
        guiValue = str2num(get(propHandle,'String'));
    else
        propHandle = handles.(['popupmenu_' propNames{i}(1:end-1)]);
        choices = get(propHandle,'String');
        guiValue = choices{get(propHandle,'Value')};
    end
    userData.properties(userData.selectedImFolder).(propNames{i})=...
        guiValue;
end

% Set properties to imFolder objects
for i=1:numel(userData.imFolders)
    set(userData.imFolders(i),userData.properties(i));
end

set(handles.figure1,'UserData',userData)
delete(handles.figure1)


% --- Executes on selection change in popupmenu_channel.
function popupmenu_channel_Callback(hObject, eventdata, handles)
userData = get(handles.figure1, 'UserData');
if get(hObject,'Value') == userData.selectedImFolder, return; end

% Retrieve handles and names  and save them
propNames = fieldnames(userData.properties(1));
for i=1:numel(propNames)
    if userData.isNumProp(i)
        propHandle = handles.(['edit_' propNames{i}(1:end-1)]);
        guiValue = str2num(get(propHandle,'String'));
    else
        propHandle = handles.(['popupmenu_' propNames{i}(1:end-1)]);
        choices = get(propHandle,'String');
        guiValue = choices{get(propHandle,'Value')};
    end
    userData.properties(userData.selectedImFolder).(propNames{i})=...
        guiValue;
end

% Update the selected imFolder path and properties
propNames = fieldnames(userData.properties(1));
userData.selectedImFolder=get(hObject,'Value'); 
set(handles.text_path, 'String', userData.imFolders(userData.selectedImFolder).imFolderPath_)
for i=1:numel(propNames)
    imFolderValue = userData.imFolders(userData.selectedImFolder).(propNames{i});
    if ~isempty(imFolderValue)
        propValue = imFolderValue;
        enableState = 'inactive';
    else
        propValue = userData.properties(userData.selectedImFolder).(propNames{i});
        enableState = 'on';
    end
    
    if userData.isNumProp(i)
        propHandle = handles.(['edit_' propNames{i}(1:end-1)]);
        guiProp = 'String';
        guiValue = propValue;
    else
        propHandle = handles.(['popupmenu_' propNames{i}(1:end-1)]);
        guiProp = 'Value';
        guiValue = find(strcmpi(propValue,get(propHandle,'String')));
        if isempty(guiValue), guiValue=1; end
    end
    set(propHandle,guiProp,guiValue,'Enable',enableState);
end


% % Display psf sigma if present
% if ~isempty(userData.imFolders(userData.selectedImFolder).psfSigma_),
%     set(handles.edit_psfSigma, 'String',...
%         userData.imFolders(userData.selectedImFolder).psfSigma_);
% else
%     set(handles.edit_psfSigma, 'String', '');
% end

set(handles.figure1,'UserData',userData)


% --- Executes when edit field is changed.
function edit_property_Callback(hObject, ~, handles)

if ~isempty(get(hObject,'String'))
    % Read property value
    propTag = get(hObject,'Tag');
    propName = [propTag(length('edit_')+1:end) '_'];
    propValue = str2double(get(hObject,'String'));

    % Test property value using the class static method
    if ~ImFolder.checkValue(propName,propValue)
        warndlg('Invalid property value','Setting Error','modal');
        set(hObject,'BackgroundColor',[1 .8 .8]);
        set(handles.popupmenu_channel, 'Enable','off');
        return
    end
end
set(hObject,'BackgroundColor',[1 1 1]);
set(handles.popupmenu_channel, 'Enable','on');


% --- Executes on selection change in popupmenu_fluorophore.
function popupmenu_fluorophore_Callback(hObject, eventdata, handles) % QZ I do not need it now.

