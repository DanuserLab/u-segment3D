function varargout = channelGUI(varargin)
% CHANNELGUI M-file for channelGUI.fig
%      CHANNELGUI, by itself, creates a new CHANNELGUI or raises the existing
%      singleton*.
%
%      H = CHANNELGUI returns the handle to a new CHANNELGUI or the handle to
%      the existing singleton*.
%
%      CHANNELGUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in CHANNELGUI.M with the given input arguments.
%
%      CHANNELGUI('Property','Value',...) creates a new CHANNELGUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before channelGUI_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to channelGUI_OpeningFcn via varargin.
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

% Edit the above text to modify the response to help channelGUI

% Last Modified by GUIDE v2.5 30-Jun-2011 15:53:11

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @channelGUI_OpeningFcn, ...
                   'gui_OutputFcn',  @channelGUI_OutputFcn, ...
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


% --- Executes just before channelGUI is made visible.
function channelGUI_OpeningFcn(hObject, ~, handles, varargin)
%
% channelGUI('mainFig', handles.figure1) - call from movieDataGUI
% channelGUI(channelArray) - call from command line
% channelGUI(..., 'modal') - call channelGUI as a modal window
%
% User Data:
% 
% userData.channels - array of channel object
% userData.selectedChannel - index of current channel
% userData.mainFig - handle of movieDataGUI
% userData.properties - structure array of properties 
% userData.propNames - list of modifiable properties
%
% userData.helpFig - handle of help window
%

set(handles.text_copyright, 'String', getLCCBCopyright())

userData = get(handles.figure1, 'UserData');

if isempty(userData), userData = struct(); end

% Choose default command line output for channelGUI
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
        % Called from movieDataGUI
        
        % Get userData.mainFig
        t = find(strcmp(varargin, 'mainFig'));
        userData.mainFig = varargin{t(end)+1};
        
        % Get userData.channels
        userData_main = get(userData.mainFig, 'UserData');
        assert(isa(userData_main.channels(1), 'Channel'), 'User-defined: No Channel object found.')
        userData.channels = userData_main.channels;
        
        % Get userData.selectedChannel
        handles_main = guidata(userData.mainFig);
        userData.selectedChannel = get(handles_main.listbox_channel, 'Value');
        
    elseif isa(varargin{1}(1), 'Channel')
        % Called from command line        
        userData.channels = varargin{1};
        userData.selectedChannel = 1;        
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
modeString = vertcat({''},Channel.getImagingModes());
set(handles.popupmenu_imageType,'String',modeString);
fluorophoreString = horzcat({''},Channel.getFluorophores());
set(handles.popupmenu_fluorophore,'String',fluorophoreString);

% Read channel initial properties and store them in a structure
propNames={'excitationWavelength_','emissionWavelength_',...
    'exposureTime_','imageType_','fluorophore_'};
userData.isNumProp = @(x) x<4;
for i=1:numel(userData.channels)
    for j=1:numel(propNames)
        userData.properties(i).(propNames{j})=...
            userData.channels(i).(propNames{j});
    end
end

% Set up pop-up menu
set(handles.popupmenu_channel, 'String', ... 
arrayfun(@(x)(['Channel ' num2str(x)]), 1:length(userData.channels), 'UniformOutput', false) )
set(handles.popupmenu_channel, 'Value', userData.selectedChannel)

% Set up channel path and properties
set(handles.figure1,'Visible','on');
set(handles.text_path, 'String', userData.channels(userData.selectedChannel).channelPath_)
for i=1:numel(propNames)
    propValue = userData.properties(userData.selectedChannel).(propNames{i});
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

% Display psf sigma if computed
if ~isempty(userData.channels(userData.selectedChannel).psfSigma_),
    set(handles.edit_psfSigma, 'String',...
        userData.channels(userData.selectedChannel).psfSigma_);
else
    set(handles.edit_psfSigma, 'String', '');
end

% Update handles structure
set(handles.figure1,'UserData',userData)
guidata(hObject, handles);

% UIWAIT makes channelGUI wait for user response (see UIRESUME)
% uiwait(handles.figure1);

% --- Outputs from this function are returned to the command line.
function varargout = channelGUI_OutputFcn(~, ~, handles) 

% Get default command line output from handles structure
varargout{1} = handles.output;

% --- Executes on selection change in popupmenu_channel.
function popupmenu_channel_Callback(hObject, ~, handles)

userData = get(handles.figure1, 'UserData');
if get(hObject,'Value') == userData.selectedChannel, return; end

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
    userData.properties(userData.selectedChannel).(propNames{i})=...
        guiValue;
end

% Update the selected channel path and properties
propNames = fieldnames(userData.properties(1));
userData.selectedChannel=get(hObject,'Value'); 
set(handles.text_path, 'String', userData.channels(userData.selectedChannel).channelPath_)
for i=1:numel(propNames)
    channelValue = userData.channels(userData.selectedChannel).(propNames{i});
    if ~isempty(channelValue)
        propValue = channelValue;
        enableState = 'inactive';
    else
        propValue = userData.properties(userData.selectedChannel).(propNames{i});
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


% Display psf sigma if present
if ~isempty(userData.channels(userData.selectedChannel).psfSigma_),
    set(handles.edit_psfSigma, 'String',...
        userData.channels(userData.selectedChannel).psfSigma_);
else
    set(handles.edit_psfSigma, 'String', '');
end

set(handles.figure1,'UserData',userData)

% --- Executes when edit field is changed.
function edit_property_Callback(hObject, ~, handles)

if ~isempty(get(hObject,'String'))
    % Read property value
    propTag = get(hObject,'Tag');
    propName = [propTag(length('edit_')+1:end) '_'];
    propValue = str2double(get(hObject,'String'));

    % Test property value using the class static method
    if ~Channel.checkValue(propName,propValue)
        warndlg('Invalid property value','Setting Error','modal');
        set(hObject,'BackgroundColor',[1 .8 .8]);
        set(handles.popupmenu_channel, 'Enable','off');
        return
    end
end
set(hObject,'BackgroundColor',[1 1 1]);
set(handles.popupmenu_channel, 'Enable','on');

% --- Executes on button press in pushbutton_cancel.
function pushbutton_cancel_Callback(~, ~, handles)

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
    userData.properties(userData.selectedChannel).(propNames{i})=...
        guiValue;
end

% Set properties to channel objects
for i=1:numel(userData.channels)
    set(userData.channels(i),userData.properties(i));
end

set(handles.figure1,'UserData',userData)
delete(handles.figure1)
        
% --- Executes on key press with focus on figure1 and none of its controls.
function figure1_KeyPressFcn(~, eventdata, handles)
if strcmp(eventdata.Key, 'return')
    pushbutton_done_Callback(handles.pushbutton_done, [], handles);
end


% --- Executes on selection change in popupmenu_fluorophore.
function popupmenu_fluorophore_Callback(hObject, eventdata, handles)

% Retrieve selected fluorophore name
props=get(hObject,{'String','Value'});
fluorophore = props{1}{props{2}};
isFluorophoreValid = ~isempty(fluorophore);
isLambdaEditable = strcmp(get(handles.edit_emissionWavelength,'Enable'),'on');
if ~isFluorophoreValid || ~isLambdaEditable,return; end

% Set the value of the emission wavelength
lambda = name2wavelength(fluorophore)*1e9;   
set(handles.edit_emissionWavelength,'String',lambda);
