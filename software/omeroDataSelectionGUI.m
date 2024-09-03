function varargout = omeroDataSelectionGUI(varargin)
% OMERODATASELECTIONGUI MATLAB code for omeroDataSelectionGUI.fig
%      OMERODATASELECTIONGUI, by itself, creates a new OMERODATASELECTIONGUI or raises the existing
%      singleton*.
%
%      H = OMERODATASELECTIONGUI returns the handle to a new OMERODATASELECTIONGUI or the handle to
%      the existing singleton*.
%
%      OMERODATASELECTIONGUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in OMERODATASELECTIONGUI.M with the given input arguments.
%
%      OMERODATASELECTIONGUI('Property','Value',...) creates a new OMERODATASELECTIONGUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before omeroDataSelectionGUI_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to omeroDataSelectionGUI_OpeningFcn via varargin.
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

% Edit the above text to modify the response to help omeroDataSelectionGUI

% Last Modified by GUIDE v2.5 29-Sep-2015 13:54:25

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
    'gui_Singleton',  gui_Singleton, ...
    'gui_OpeningFcn', @omeroDataSelectionGUI_OpeningFcn, ...
    'gui_OutputFcn',  @omeroDataSelectionGUI_OutputFcn, ...
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


% --- Executes just before omeroDataSelectionGUI is made visible.
function omeroDataSelectionGUI_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to omeroDataSelectionGUI (see VARARGIN)

ip = inputParser;
ip.addRequired('hObject',@ishandle);
ip.addRequired('eventdata',@(x) isstruct(x) || isempty(x));
ip.addRequired('handles',@isstruct);
ip.addOptional('MD',[],@(x) isa(x,'MovieData'));
ip.addParamValue('mainFig', -1, @ishandle);
ip.parse(hObject,eventdata,handles,varargin{:})

% Store input
userData = get(handles.figure1, 'UserData');
if isempty(userData), userData = struct(); end
userData.mainFig=ip.Results.mainFig;

set(handles.text_copyright, 'String', getLCCBCopyright())
global session

% Retrieve user
adminService = session.getAdminService();
userId = adminService.getEventContext().userId;
groupId = adminService.getEventContext().groupId;
userData.experimenter = session.getAdminService().getExperimenter(userId);

% Populate drop-down menu for groups the user is member of
user = omero.model.ExperimenterI(userId, false);
groupIds1 = toMatlabList(adminService.getLeaderOfGroupIds(user));
groupIds2 = toMatlabList(adminService.getMemberOfGroupIds(user));
groupIds = [groupIds1 groupIds2];
systemGroupId = adminService.getSecurityRoles().systemGroupId;
userGroupId = adminService.getSecurityRoles().userGroupId;
try
    guestGroupId = adminService.getSecurityRoles().guestGroupId;
catch
    guestGroupId = -1;
end
% Filter out system groups and update popup menu
groupIds(ismember(groupIds,...
    [systemGroupId userGroupId guestGroupId])) = [];
groupNames = arrayfun(@(x) char(adminService.getGroup(x).getName().getValue),...
    groupIds, 'UniformOutput', false);
set(handles.popupmenu_group, 'String', groupNames, 'UserData', groupIds,...
    'Value', find(groupId == groupIds));

set(hObject, 'UserData', userData);

refreshGroupList(hObject, [], handles);

% Choose default command line output for omeroDataSelectionGUI
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);


% --- Outputs from this function are returned to the command line.
function varargout = omeroDataSelectionGUI_OutputFcn(hObject, eventdata, handles)
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;

% --- Executes on selection change in popupmenu_user.
function refreshGroupList(hObject, eventdata, handles)
global session

% Retrieve the selected group identifier
props = get(handles.popupmenu_group, {'Value', 'UserData'});
groupId = props{2}(props{1});

userData = get(handles.figure1, 'UserData');

adminService = session.getAdminService();
userId = adminService.getEventContext().userId;
group = session.getAdminService().getGroup(groupId);       
userData.groupPermissions = group.getDetails().getPermissions();
set(handles.figure1, 'UserData', userData);

display_username = @(x) [char(x.getFirstName().getValue())...
        ' ' char(x.getLastName().getValue())...
        ' (' char(x.getOmeName().getValue()) ')'];
if userData.groupPermissions.isGroupRead(),
    experimenters = toMatlabList(group.linkedExperimenterList);
    ids = arrayfun(@(x) x.getId().getValue(), experimenters);
    names = arrayfun(display_username, experimenters, 'UniformOutput', false);
    set(handles.popupmenu_user, 'Enable', 'on', 'String', names,...
        'UserData', ids, 'Value', find(ids == userId));
else
    name = display_username(userData.experimenter);
    set(handles.popupmenu_user, 'Enable', 'off', 'String', name,...
        'UserData', userId, 'Value', 1);
end

set(handles.figure1, 'UserData', userData);
refreshUserList(hObject, [], handles);


% --- Executes on selection change in popupmenu_user.
function refreshUserList(hObject, eventdata, handles)

global session

% Retrieve the selected user and group identifier
props = get(handles.popupmenu_group, {'Value', 'UserData'});
groupId = props{2}(props{1});
props = get(handles.popupmenu_user, {'Value', 'UserData'});
userId = props{2}(props{1});

userData = get(handles.figure1, 'UserData');

if userId ~= session.getAdminService.getEventContext().userId && ...
        ~userData.groupPermissions.isGroupAnnotate(),
    set(handles.pushbutton_load_images, 'Enable', 'off');
    set(handles.pushbutton_load_dataset, 'Enable', 'off');
else
    set(handles.pushbutton_load_images, 'Enable', 'on');
    set(handles.pushbutton_load_dataset, 'Enable', 'on');
end

% List projects and orphaned datasets
[projects, orphanedDatasets] = getProjects(...
    session, 'owner', userId, 'group', groupId); 
projects = sortById(projects);
orphanedDatasets = sortById(orphanedDatasets);


projectNames = arrayfun(@(x) char(x.getName().getValue()), projects,...
    'UniformOutput', false);
projectNames = [{'none'}; projectNames];
set(handles.popupmenu_project, 'Value', 1, 'String', projectNames,...
    'UserData', projects);

% Save orphaned datasets
userData = get(handles.figure1, 'UserData');
if isempty(userData), userData = struct(); end
userData.orphaned_datasets = orphanedDatasets;
set(handles.figure1, 'UserData', userData);

refreshDatasetList(hObject, [], handles);

function refreshImageList(handles)
global session
% Read image list from selected dataset
p_props = get(handles.popupmenu_project, {'Value', 'UserData'});
d_props = get(handles.popupmenu_dataset, {'Value', 'UserData'});
if d_props{1} == 1
    images = [];
elseif  p_props{1} == 1 && d_props{1} == numel(d_props{2}) + 2;
    % Retrieve orphaned images
    orphanQuery = ['select img from Image as img '...
        'left outer join fetch img.details.owner '...
        'left outer join fetch img.pixels as pix '...
        'left outer join fetch pix.pixelsType as pt '...
        'where not exists (select obl from '...'
        'DatasetImageLink as obl where obl.child = img.id) '...
        'and not exists (select ws from WellSample as '...
        'ws where ws.image = img.id)'...
        ' and img.details.owner.id = :userID'];
    parameters = omero.sys.ParametersI();
    u_props = get(handles.popupmenu_user, {'Value', 'UserData'});
    parameters.addLong('userID', u_props{2}(u_props{1}));
    images = session.getQueryService.findAllByQuery(orphanQuery, parameters);
    images = toMatlabList(images);
    
else
    datasetId = d_props{2}(d_props{1}-1).getId.getValue;
    images = getImages(session, 'dataset', datasetId);
end

% Update image list
images = sortById(images);
imageNames = arrayfun(@(x) char(x.getName().getValue()), images,...
    'UniformOutput', false);
set(handles.listbox_images, 'Value', 1, 'String', imageNames,...
    'UserData', images);

% --- Executes on selection change in popupmenu_project.
function refreshDatasetList(hObject, eventdata, handles)

% Read dataset list from selected project
props = get(handles.popupmenu_project, {'Value', 'UserData'});
if props{1} == 1,
    userData = get(handles.figure1, 'UserData');
    datasets = userData.orphaned_datasets;
else
    datasets = toMatlabList(props{2}(props{1} - 1).linkedDatasetList);
    datasets = sortById(datasets);
end

% Update datasets drop down menu list
datasetNames = arrayfun(@(x) char(x.getName().getValue()), datasets,...
    'UniformOutput', false);
datasetNames = [{'none'}; datasetNames];
if props{1} == 1,
    datasetNames = [datasetNames; {'Orphaned images'}];
end
set(handles.popupmenu_dataset, 'Value', 1, 'String', datasetNames,...
    'UserData', datasets);
refreshImageList(handles)

% --- Executes on selection change in popupmenu_dataset.
function popupmenu_dataset_Callback(hObject, eventdata, handles)
refreshImageList(handles)

% --- Executes on button press in pushbutton_load_images.
function pushbutton_load_images_Callback(hObject, eventdata, handles)

props = get(handles.listbox_images, {'Value', 'UserData'});
if isempty(props{1}), return; end
imageIDs = arrayfun(@(x) x.getId().getValue(), props{2}(props{1}));

% If derivated from movie selection window, only load new movies
userData = get(handles.figure1, 'UserData');
if isempty(userData), userData = struct(); end
if ishandle(userData.mainFig),
    userData=get(handles.figure1,'UserData');
    userData_main = get(userData.mainFig, 'UserData');
    omeroMovies = arrayfun(@isOmero, userData_main.MD);
    existingIDs = arrayfun(@(x) x.getOmeroId(), userData_main.MD(omeroMovies));
    imageIDs = setdiff(imageIDs, existingIDs);
end

% Return if empty list
if isempty(imageIDs),
    errordlg('All selected images have already been loaded', 'Error', 'modal');
    return
end

% Load images from OMERO
global session
MD = getOmeroMovies(session, imageIDs);

% Update movie selector interface
if ishandle(userData.mainFig),
    % Append  MovieData object to movie selector panel
    userData_main.MD = horzcat(userData_main.MD, MD);
    set(userData.mainFig, 'UserData', userData_main)
    movieSelectorGUI('refreshDisplay', userData.mainFig,...
        eventdata, guidata(userData.mainFig))
end

delete(handles.figure1);

% --- Executes on button press in pushbutton_cancel.
function pushbutton_cancel_Callback(hObject, eventdata, handles)

delete(handles.figure1);


% --- Executes on button press in pushbutton_load_dataset.
function pushbutton_load_dataset_Callback(hObject, eventdata, handles)

props = get(handles.popupmenu_dataset, {'Value', 'UserData'});
if props{1} == 1, return; end
datasetIDs = props{2}(props{1}-1).getId().getValue();

% If derivated from movie selection window, only load new movies
userData = get(handles.figure1, 'UserData');
if isempty(userData), userData = struct(); end
if ishandle(userData.mainFig),
    userData=get(handles.figure1,'UserData');
    userData_main = get(userData.mainFig, 'UserData');
    omeroLists = arrayfun(@isOmero, userData_main.ML);
    existingIDs = arrayfun(@(x) x.getOmeroId(), userData_main.ML(omeroLists));
    datasetIDs = setdiff(datasetIDs, existingIDs);
end

% Return if empty list
if isempty(datasetIDs),
    errordlg('All selected datasets have already been loaded', 'Error', 'modal');
    return
end

% Load images from OMERO
global session
ML = getOmeroLists(session, datasetIDs);

% Update movie selector interface
if ishandle(userData.mainFig),
    % Append  MovieList object to movie selector panel
    userData_main.ML = horzcat(userData_main.ML, ML);
    
    % Append new MovieData objects to movie selector panel
    movieList = arrayfun(@getFullPath, userData_main.MD, 'Unif', false);
    if ~isempty(movieList),
        index = cellfun(@(x) ~ismember(x.getFullPath(), movieList),...
            ML.getMovies());
    else
        index = true(numel(ML.getMovies), 1);
    end
    
    if any(index)
        newMovies = ML.getMovies();
        newMovies = newMovies(index);
        userData_main.MD = horzcat(userData_main.MD, newMovies{:});
    else
        warndlg('All images containedin dataset has already been loaded',...
            'Warning','modal');
    end
    
    set(userData.mainFig, 'UserData', userData_main)
    movieSelectorGUI('refreshDisplay', userData.mainFig,...
        eventdata, guidata(userData.mainFig))
end

delete(handles.figure1);

function objects = sortById(objects)

objectIds = arrayfun(@(x) x.getId().getValue(), objects);
[~, sortindex] = sort(objectIds);
objects =objects(sortindex);


% --- Executes on selection change in popupmenu_project.
function popupmenu_project_Callback(hObject, eventdata, handles)
refreshDatasetList(hObject, eventdata, handles);


% --- Executes on selection change in popupmenu_group.
function popupmenu_group_Callback(hObject, eventdata, handles)
refreshGroupList(hObject, eventdata, handles);


% --- Executes on selection change in popupmenu_user.
function popupmenu_user_Callback(hObject, eventdata, handles)
refreshUserList(hObject, eventdata, handles);
