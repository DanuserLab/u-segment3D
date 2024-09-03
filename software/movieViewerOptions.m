function optionsFig = movieViewerOptions(mainFig)
%GRAPHVIEWER creates a graphical interface to control the movie options
%
% This function creates a list of checkboxes for all graph processes which
% output can be displayed in a standalone figure. It is called by
% movieViewer.
% 
% Input 
%
%   mainFig - the handle of the calling figure.
%
% Output:
%   
%   optionsFig - the handle of the movie options interface
%
% See also: graphViewer, movieViewerOptions
%
% Sebastien Besson, Nov 2012
% Andrew R. Jamieson - Modified Feb 2017
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

% Check existence of viewer
h=findobj(0,'Name','Movie options');
if ~isempty(h), delete(h); end
optionsFig=figure('Name','Movie options','Position',[0 0 200 200],...
    'NumberTitle','off','Tag','figure1','Toolbar','none','MenuBar','none',...
    'Color',get(0,'defaultUicontrolBackgroundColor'),'Resize','off');

userData = get(mainFig, 'UserData');

%% Image options panel creation
imagePanel = uibuttongroup(gcf,'Position',[0 0 1/2 1],...
    'Title','Image options','BackgroundColor',get(0,'defaultUicontrolBackgroundColor'),...
    'Units','pixels','Tag','uipanel_image');

% Timestamp
hPosition=10;
if isempty(userData.MO.timeInterval_),
    timeStampStatus = 'off';
else
    timeStampStatus = 'on';
end
uicontrol(imagePanel,'Style','checkbox',...
    'Position',[10 hPosition 200 20],'Tag','checkbox_timeStamp',...
    'String',' Time stamp','HorizontalAlignment','left',...
    'Enable',timeStampStatus,'Callback',@(h,event) setTimeStamp(guidata(h)));
uicontrol(imagePanel,'Style','popupmenu','Position',[130 hPosition 120 20],...
    'String',{'NorthEast', 'SouthEast', 'SouthWest', 'NorthWest'},'Value',4,...
    'Tag','popupmenu_timeStampLocation','Enable',timeStampStatus,...
    'Callback',@(h,event) setTimeStamp(guidata(h)));

% Scalebar
hPosition=hPosition+30;
if isempty(userData.MO.pixelSize_), scBarstatus = 'off'; else scBarstatus = 'on'; end
uicontrol(imagePanel,'Style','edit','Position',[30 hPosition 50 20],...
    'String','1','BackgroundColor','white','Tag','edit_imageScaleBar',...
    'Enable',scBarstatus,...
    'Callback',@(h,event) setScaleBar(guidata(h),'imageScaleBar'));
uicontrol(imagePanel,'Style','text','Position',[85 hPosition-2 70 20],...
    'String','microns','HorizontalAlignment','left');
uicontrol(imagePanel,'Style','checkbox',...
    'Position',[150 hPosition 100 20],'Tag','checkbox_imageScaleBarLabel',...
    'String',' Show label','HorizontalAlignment','left',...
    'Enable',scBarstatus,...
    'Callback',@(h,event) setScaleBar(guidata(h),'imageScaleBar'));

hPosition=hPosition+30;
uicontrol(imagePanel,'Style','checkbox',...
    'Position',[20 hPosition 200 20],'Tag','checkbox_imageScaleBar',...
    'String',' Show scalebar','HorizontalAlignment','left',...
    'Enable',scBarstatus,...
    'Callback',@(h,event) setScaleBar(guidata(h),'imageScaleBar'));
uicontrol(imagePanel,'Style','popupmenu','Position',[130 hPosition 120 20],...
    'String',{'NorthEast', 'SouthEast', 'SouthWest', 'NorthWest'},'Value',3,...
    'Tag','popupmenu_imageScaleBarLocation','Enable',scBarstatus,...
    'Callback',@(h,event) setScaleBar(guidata(h),'imageScaleBar'));

hPosition=hPosition+20;
uicontrol(imagePanel,'Style','text',...
    'Position',[10 hPosition 200 20],'Tag','text_scalebar',...
    'String','Scalebar','HorizontalAlignment','left','FontWeight','bold');

hPosition=hPosition+30;
uicontrol(imagePanel,'Style','pushbutton',...
    'Position',[20 hPosition 300 20],'Tag','pushbutton_scaleFactor',...
    'String','Calibrate the ratio map unit','HorizontalAlignment','left',...
    'Callback',@(h,event) calibrateRatio(guidata(h)));

hPosition=hPosition+25;
uicontrol(imagePanel,'Style','text',...
    'Position',[20 hPosition 130 20],'Tag','text_imageChannelScaleFactor',...
    'String','Channel Scaling','HorizontalAlignment','left');
uicontrol(imagePanel,'Style','edit','Position',[130 hPosition 25 20],...
    'String','1','BackgroundColor','white','Tag','edit_imageChannelScaleFactor_R',...
    'Callback',@(h,event) setScaleFactorPerChannel(guidata(h)));    
uicontrol(imagePanel,'Style','edit','Position',[130+30 hPosition 25 20],...
    'String','1','BackgroundColor','white','Tag','edit_imageChannelScaleFactor_G',...
    'Callback',@(h,event) setScaleFactorPerChannel(guidata(h)));
uicontrol(imagePanel,'Style','edit','Position',[130+60 hPosition 25 20],...
    'String','1','BackgroundColor','white','Tag','edit_imageChannelScaleFactor_B',...
    'Callback',@(h,event) setScaleFactorPerChannel(guidata(h)));


hPosition=hPosition+25;
stepx = 30;
xpos_ = 0;
uicontrol(imagePanel,'Style','text',...
    'Position',[20 hPosition 130 20],'Tag','text_imageChannelContrast',...
    'String','RGBx[low high]','HorizontalAlignment','left');
uicontrol(imagePanel,'Style','edit','Position',[130+xpos_ hPosition 25 20],...
    'String','0','BackgroundColor','white','Tag','edit_imageChannelContrast_R_low',...
    'Callback',@(h,event) setScaleFactorPerChannel(guidata(h)));    
xpos_ = xpos_ + stepx;
uicontrol(imagePanel,'Style','edit','Position',[130+xpos_ hPosition 25 20],...
    'String','1','BackgroundColor','white','Tag','edit_imageChannelContrast_R_high',...
    'Callback',@(h,event) setScaleFactorPerChannel(guidata(h)));    
xpos_ = xpos_ + stepx+10;
uicontrol(imagePanel,'Style','edit','Position',[130+xpos_ hPosition 25 20],...
    'String','0','BackgroundColor','white','Tag','edit_imageChannelContrast_G_low',...
    'Callback',@(h,event) setScaleFactorPerChannel(guidata(h)));    
xpos_ = xpos_ + stepx;
uicontrol(imagePanel,'Style','edit','Position',[130+xpos_ hPosition 25 20],...
    'String','1','BackgroundColor','white','Tag','edit_imageChannelContrast_G_high',...
    'Callback',@(h,event) setScaleFactorPerChannel(guidata(h)));    
xpos_ = xpos_ + stepx+10;
uicontrol(imagePanel,'Style','edit','Position',[130+xpos_ hPosition 25 20],...
    'String','0','BackgroundColor','white','Tag','edit_imageChannelContrast_B_low',...
    'Callback',@(h,event) setScaleFactorPerChannel(guidata(h)));    
xpos_ = xpos_ + stepx;
uicontrol(imagePanel,'Style','edit','Position',[130+xpos_ hPosition 25 20],...
    'String','1','BackgroundColor','white','Tag','edit_imageChannelContrast_B_high',...
    'Callback',@(h,event) setScaleFactorPerChannel(guidata(h)));    





hPosition=hPosition+25;
uicontrol(imagePanel,'Style','text',...
    'Position',[20 hPosition 100 20],'Tag','text_imageScaleFactor',...
    'String','Scaling factor','HorizontalAlignment','left');
uicontrol(imagePanel,'Style','edit','Position',[130 hPosition 70 20],...
    'String','1','BackgroundColor','white','Tag','edit_imageScaleFactor',...
    'Callback',@(h,event) setScaleFactor(guidata(h)));


% Colormap control
hPosition=hPosition+30;
uicontrol(imagePanel,'Style','text','Position',[20 hPosition-2 100 20],...
    'String','Color limits','HorizontalAlignment','left');
uicontrol(imagePanel,'Style','edit','Position',[130 hPosition 70 20],...
    'String','','BackgroundColor','white','Tag','edit_cmin',...
    'Callback',@(h,event) setCLim(guidata(h)));
uicontrol(imagePanel,'Style','edit','Position',[200 hPosition 70 20],...
    'String','','BackgroundColor','white','Tag','edit_cmax',...
    'Callback',@(h,event) setCLim(guidata(h)));

hPosition=hPosition+30;
uicontrol(imagePanel,'Style','text','Position',[20 hPosition-2 80 20],...
    'String','Colormap','HorizontalAlignment','left');
uicontrol(imagePanel,'Style','popupmenu',...
    'Position',[130 hPosition 120 20],'Tag','popupmenu_colormap',...
    'String',{'Gray','Jet','Parula','HSV'},'Value',1,...
    'HorizontalAlignment','left','Callback',@(h,event) setColormap(guidata(h)));

% Colormap inversion
uicontrol(imagePanel,'Style','checkbox',...
    'Position',[255 hPosition 120 20],'Tag','checkbox_invertcolormap',...
    'Value',0, 'String',' Invert', ...
    'HorizontalAlignment','left','Callback',@(h,event) setColormapInvert(guidata(h)));

% Colorbar 
hPosition=hPosition+30;
uicontrol(imagePanel,'Style','checkbox',...
    'Position',[10 hPosition 120 20],'Tag','checkbox_colorbar',...
    'String',' Colorbar','HorizontalAlignment','left',...
    'Callback',@(h,event) setColorbar(guidata(h)));
%findclass(findpackage('scribe'),'colorbar');
locations = ImageDisplay.getColorBarLocations();
uicontrol(imagePanel,'Style','popupmenu','String',locations,...
    'Position',[130 hPosition 120 20],'Tag','popupmenu_colorbarLocation',...
    'HorizontalAlignment','left','Callback',@(h,event) setColorbar(guidata(h)));

%% Overlay options creation
overlayPanel = uipanel(gcf,'Position',[1/2 0 1/2 1],...
    'Title','Overlay options','BackgroundColor',get(0,'defaultUicontrolBackgroundColor'),...
    'Units','pixels','Tag','uipanel_overlay');

hPosition=10;
% First create overlay option (vectorField)
if isempty(userData.MO.pixelSize_) || isempty(userData.MO.timeInterval_),
    scaleBarStatus = 'off';
else
    scaleBarStatus = 'on';
end
uicontrol(overlayPanel,'Style','text','Position',[20 hPosition-2 100 20],...
    'String','Color limits','HorizontalAlignment','left');
uicontrol(overlayPanel,'Style','edit','Position',[150 hPosition 50 20],...
    'String','','BackgroundColor','white','Tag','edit_vectorCmin',...
    'Callback',@(h,event) userData.redrawOverlaysFcn());
uicontrol(overlayPanel,'Style','edit','Position',[200 hPosition 50 20],...
    'String','','BackgroundColor','white','Tag','edit_vectorCmax',...
    'Callback',@(h,event) userData.redrawOverlaysFcn());

hPosition=hPosition+30;
uicontrol(overlayPanel,'Style','edit','Position',[30 hPosition 50 20],...
    'String','1000','BackgroundColor','white','Tag','edit_vectorFieldScaleBar',...
    'Enable',scaleBarStatus,...
    'Callback',@(h,event) setScaleBar(guidata(h),'vectorFieldScaleBar'));
uicontrol(overlayPanel,'Style','text','Position',[85 hPosition-2 70 20],...
    'String','nm/min','HorizontalAlignment','left');
uicontrol(overlayPanel,'Style','checkbox',...
    'Position',[150 hPosition 100 20],'Tag','checkbox_vectorFieldScaleBarLabel',...
    'String',' Show label','HorizontalAlignment','left',...
    'Enable',scaleBarStatus,...
    'Callback',@(h,event) setScaleBar(guidata(h),'vectorFieldScaleBar'));

hPosition=hPosition+30;
uicontrol(overlayPanel,'Style','checkbox',...
    'Position',[20 hPosition 100 20],'Tag','checkbox_vectorFieldScaleBar',...
    'String',' Scalebar','HorizontalAlignment','left',...
    'Enable',scaleBarStatus,...
    'Callback',@(h,event) setScaleBar(guidata(h),'vectorFieldScaleBar'));
uicontrol(overlayPanel,'Style','popupmenu','Position',[130 hPosition 120 20],...
    'String',{'NorthEast', 'SouthEast', 'SouthWest', 'NorthWest'},'Value',3,...
    'Tag','popupmenu_vectorFieldScaleBarLocation','Enable',scaleBarStatus,...
    'Callback',@(h,event) setScaleBar(guidata(h),'vectorFieldScaleBar'));

hPosition=hPosition+30;
uicontrol(overlayPanel,'Style','text',...
    'Position',[20 hPosition 100 20],'Tag','text_vectorFieldScale',...
    'String',' Display scale','HorizontalAlignment','left');
uicontrol(overlayPanel,'Style','edit','Position',[120 hPosition 50 20],...
    'String','1','BackgroundColor','white','Tag','edit_vectorFieldScale',...
    'Callback',@(h,event) setVectorScaleFactor(guidata(h)));

hPosition=hPosition+20;
uicontrol(overlayPanel,'Style','text',...
    'Position',[10 hPosition 200 20],'Tag','text_vectorFieldOptions',...
    'String','Vector field options','HorizontalAlignment','left','FontWeight','bold');

% Track options
hPosition=hPosition+30;
uicontrol(overlayPanel,'Style','text',...
    'Position',[20 hPosition 100 20],'Tag','text_dragtailLength',...
    'String',' Dragtail length','HorizontalAlignment','left');
uicontrol(overlayPanel,'Style','edit','Position',[120 hPosition 50 20],...
    'String','10','BackgroundColor','white','Tag','edit_dragtailLength',...
    'Callback',@(h,event) userData.redrawOverlaysFcn());

hPosition=hPosition+20;
uicontrol(overlayPanel,'Style','checkbox',...
    'Position',[20 hPosition 150 20],'Tag','checkbox_markMergeSplit',...
    'String',' Mark merge/split','HorizontalAlignment','left',...
    'Callback',@(h,event) userData.redrawOverlaysFcn());

hPosition=hPosition+20;
uicontrol(overlayPanel,'Style','checkbox',...
    'Position',[20 hPosition 150 20],'Tag','checkbox_showLabel',...
    'String',' Show track number','HorizontalAlignment','left',...
    'Callback',@(h,event) userData.redrawOverlaysFcn());

hPosition=hPosition+20;
uicontrol(overlayPanel,'Style','text',...
    'Position',[10 hPosition 200 20],'Tag','text_trackOptions',...
    'String','Track options','HorizontalAlignment','left','FontWeight','bold');

% Windows options
hPosition=hPosition+30;
uicontrol(overlayPanel,'Style','text',...
    'Position',[20 hPosition 150 20],'Tag','text_ProjectionAxis',...
    'String',' Projection Axis','HorizontalAlignment','left');
uicontrol(overlayPanel,'Style','popupmenu','String', {'Z', 'Y', 'X','three'},'Value',1,...
    'Position',[135 hPosition 35 20],...
    'BackgroundColor','white','Tag','popupmenu_ProjectionAxis',...
    'Callback', @(h,event) userData.redrawOverlaysFcn());

hPosition=hPosition+30;
uicontrol(overlayPanel,'Style','text',...
    'Position',[20 hPosition 100 20],'Tag','text_faceAlpha',...
    'String',' Alpha value','HorizontalAlignment','left');
uicontrol(overlayPanel,'Style','edit','Position',[120 hPosition 50 20],...
    'String','.3','BackgroundColor','white','Tag','edit_faceAlpha',...
    'Callback',@(h,event) userData.redrawOverlaysFcn());


hPosition=hPosition+20;
uicontrol(overlayPanel,'Style','text',...
    'Position',[10 hPosition 200 20],'Tag','text_windowsOptions',...
    'String','Windows options','HorizontalAlignment','left','FontWeight','bold');



%% Get image/overlay panel size and resize them
imagePanelSize = getPanelSize(imagePanel);
overlayPanelSize = getPanelSize(overlayPanel);
panelsLength = max(500,imagePanelSize(1)+overlayPanelSize(1));
panelsHeight = max([imagePanelSize(2),overlayPanelSize(2)]);

% Resize panel
if ishandle(imagePanel)
    set(imagePanel,'Position',[10 panelsHeight-imagePanelSize(2)+10 ...
        imagePanelSize(1) imagePanelSize(2)])
end
if ishandle(overlayPanel)
    set(overlayPanel,'Position',[imagePanelSize(1)+10 panelsHeight-overlayPanelSize(2)+10 ...
        overlayPanelSize(1) overlayPanelSize(2)]);
end

%% Resize panels and figure
sz=get(0,'ScreenSize');
maxWidth = panelsLength+20;
maxHeight = panelsHeight;
figWidth = min(maxWidth,.75*sz(3));
figHeight=min(maxHeight,.75*sz(4));
set(optionsFig,'Position',[sz(3)/50 10 figWidth figHeight+20]);

% Update handles structure and attach it to the main figure
handles = guihandles(optionsFig);
guidata(handles.figure1, handles);

userData.mainFig = mainFig;
userData.setImageOptions = @(h, method) setImageOptions(handles, h, method);
userData.getOverlayOptions = @(h,event) getOverlayOptions(handles);
userData.setOverlayOptions = @(method) setOverlayOptions(handles, method);
userData.setProjectionAxis3D = @(projectionAxis3D) setProjectionAxis3D(handles, projectionAxis3D);

set(handles.figure1,'UserData',userData);

function size = getPanelSize(hPanel)
if ~ishandle(hPanel), size=[0 0]; return; end
a=get(get(hPanel,'Children'),'Position');
P=vertcat(a{:});
size = [max(P(:,1)+P(:,3))+10 max(P(:,2)+P(:,4))+20];


function setScaleBar(handles,type)
% Remove existing scalebar of given type
h=findobj('Tag',type);
if ~isempty(h), delete(h); end

% If checked, adds a new scalebar using the width as a label input
userData=get(handles.figure1,'UserData');
if ~get(handles.(['checkbox_' type]),'Value'), return; end
userData.getFigure('Movie');

scale = str2double(get(handles.(['edit_' type]),'String'));
if strcmp(type,'imageScaleBar')
    width = scale *1000/userData.MO.pixelSize_;
    label = [num2str(scale) ' \mum'];
else
    displayScale = str2double(get(handles.edit_vectorFieldScale,'String'));
    width = scale*displayScale/(userData.MO.pixelSize_/userData.MO.timeInterval_*60);
    label= [num2str(scale) ' nm/min'];
end
if ~get(handles.(['checkbox_' type 'Label']),'Value'), label=''; end
props=get(handles.(['popupmenu_' type 'Location']),{'String','Value'});
location=props{1}{props{2}};
hScaleBar = plotScaleBar(width,'Label',label,'Location',location);
set(hScaleBar,'Tag',type);

function setTimeStamp(handles)
% Remove existing timestamp of given type
h=findobj('Tag','timeStamp');
if ~isempty(h), delete(h); end

% If checked, adds a new scalebar using the width as a label input
userData=get(handles.figure1,'UserData');
if ~get(handles.checkbox_timeStamp,'Value'), return; end
userData.getFigure('Movie');

mainhandles = guidata(userData.mainFig);
frameNr=get(mainhandles.slider_frame,'Value');
time= (frameNr-1)*userData.MO.timeInterval_;
p=sec2struct(time);
props=get(handles.popupmenu_timeStampLocation,{'String','Value'});
location=props{1}{props{2}};
hTimeStamp = plotTimeStamp(p.str,'Location',location);
set(hTimeStamp,'Tag','timeStamp');

function setCLim(handles)
clim=[str2double(get(handles.edit_cmin,'String')) ...
    str2double(get(handles.edit_cmax,'String'))];

userData = get(handles.figure1,'UserData');
userData.redrawImageFcn(handles,'CLim',clim)

function setScaleFactor(handles)
% grab overall scale factor
scaleFactor=str2double(get(handles.edit_imageScaleFactor,'String'));
userData = get(handles.figure1,'UserData');
userData.redrawImageFcn(handles,'ScaleFactor',scaleFactor)

function setScaleFactorPerChannel(handles)
% grab per channel scale factors
ScaleFactorRGB = [1 1 1 0 1 0 1 0 1];

ScaleFactorRGB(1) = str2double(get(handles.edit_imageChannelScaleFactor_R,'String'));
ScaleFactorRGB(2) = str2double(get(handles.edit_imageChannelScaleFactor_G,'String'));
ScaleFactorRGB(3) = str2double(get(handles.edit_imageChannelScaleFactor_B,'String'));

ScaleFactorRGB(5) = str2double(get(handles.edit_imageChannelContrast_R_high,'String'));
ScaleFactorRGB(4) = str2double(get(handles.edit_imageChannelContrast_R_low,'String'));

ScaleFactorRGB(7) = str2double(get(handles.edit_imageChannelContrast_G_high,'String'));
ScaleFactorRGB(6) = str2double(get(handles.edit_imageChannelContrast_G_low,'String'));

ScaleFactorRGB(9) = str2double(get(handles.edit_imageChannelContrast_B_high,'String'));
ScaleFactorRGB(8) = str2double(get(handles.edit_imageChannelContrast_B_low,'String'));

userData = get(handles.figure1,'UserData');
userData.redrawImageFcn(handles,'ScaleFactorRGB',ScaleFactorRGB)


function calibrateRatio(handles)

% Make sure a movie is drawn
userData = get(handles.figure1, 'UserData');
h = findobj(0, '-regexp', 'Name', '^Movie$');
if isempty(h), userData.redrawImageFcn(handles); end

% Retrieve the handle of the axes containing the image
hImage = findobj(h, 'Type', 'image', '-and', '-regexp', 'Tag', 'process','-or','Tag','channels');
hAxes = get(hImage, 'Parent');

% Allow use to draw polygon and retrieve mask once it is double-clicked
p = impoly(hAxes);
wait(p);
mask = createMask(p);
p.delete();

% Calculate mean value of unscaled image within the mask
scaleFactor = str2double(get(handles.edit_imageScaleFactor, 'String'));
imData = get(hImage, 'CData') * scaleFactor;
ratio = nanmean(imData(mask));

% Set the color limit and scale factor of the image
clim=[str2double(get(handles.edit_cmin, 'String')) ...
    str2double(get(handles.edit_cmax, 'String'))];
clim(1) = ratio;
scaleFactor = ratio;
userData.redrawImageFcn(handles, 'CLim', clim, 'ScaleFactor', scaleFactor);

function setVectorScaleFactor(handles)

userData = get(handles.figure1,'UserData');
userData.redrawOverlaysFcn();

% Reset the vector field scalebar
if get(handles.checkbox_vectorFieldScaleBar,'Value'),
    setScaleBar(handles,'vectorFieldScaleBar');
end

function setColormap(handles)
allCmap=get(handles.popupmenu_colormap,'String');
selectedCmap = get(handles.popupmenu_colormap,'Value');

userData = get(handles.figure1,'UserData');
userData.redrawImageFcn('Colormap', allCmap{selectedCmap})

function setColormapInvert(handles)
invertCmap = get(handles.checkbox_invertcolormap,'Value');
userData = get(handles.figure1,'UserData');
userData.redrawImageFcn('invertColormap', logical(invertCmap))

function setColorbar(handles)
cbar=get(handles.checkbox_colorbar,'Value');
props = get(handles.popupmenu_colorbarLocation,{'String','Value'});
if cbar, cbarStatus='on'; else cbarStatus='off'; end

userData = get(handles.figure1,'UserData');
userData.redrawImageFcn(handles,'Colorbar',cbarStatus,'ColorbarLocation',props{1}{props{2}})

function setImageOptions(handles, drawFig, displayMethod)

% Set the color limits properties
clim=displayMethod.CLim;
if isempty(clim)
    hAxes=findobj(drawFig,'Type','axes','-not','Tag','Colorbar');
    clim=get(hAxes,'Clim');
end
if ~isempty(clim)
    set(handles.edit_cmin,'Enable','on','String',clim(1));
    set(handles.edit_cmax,'Enable','on','String',clim(2));
end

set(handles.edit_imageScaleFactor, 'Enable', 'on',...
    'String',displayMethod.ScaleFactor);

% Set the colorbar properties
cbar=displayMethod.Colorbar;
cbarLocation = find(strcmpi(displayMethod.ColorbarLocation,get(handles.popupmenu_colorbarLocation,'String')));
set(handles.checkbox_colorbar,'Value',strcmp(cbar,'on'));
set(handles.popupmenu_colorbarLocation,'Enable',cbar,'Value',cbarLocation);

% Set the colormap properties
cmap=displayMethod.Colormap;
colormaps = {'Gray','Jet','Parula','HSV'};
iCmap = find(strcmpi(cmap,colormaps),1);

if isempty(iCmap), 
    set(handles.popupmenu_colormap,'String','Custom',...
        'Value',1,'Enable','off');
else
    set(handles.popupmenu_colormap,'String',colormaps,...
        'Value',iCmap,'Enable','on');
end

% Reset the scaleBar
setScaleBar(handles,'imageScaleBar');
setTimeStamp(handles);
% setScaleFactorPerChannel(handles);



function setOverlayOptions(handles, displayMethod)

% Set the color limits properties
if isa(displayMethod,'VectorFieldDisplay') && ~isempty(displayMethod.CLim)
    set(handles.edit_vectorCmin,'String',displayMethod.CLim(1));
    set(handles.edit_vectorCmax,'String',displayMethod.CLim(2));
end

function setProjectionAxis3D(handles, projectionAxis3D)
    newValue = find(ismember(handles.popupmenu_ProjectionAxis.String, projectionAxis3D));
    set(handles.popupmenu_ProjectionAxis,'Value', newValue);

function options = getOverlayOptions(handles)

% Read various options
projectionAxis3D = handles.popupmenu_ProjectionAxis.String(get(handles.popupmenu_ProjectionAxis,'Value'));
vectorScale = str2double(get(handles.edit_vectorFieldScale,'String'));
dragtailLength = str2double(get(handles.edit_dragtailLength,'String'));    
showLabel = get(handles.checkbox_showLabel,'Value');
markMergeSplit = get(handles.checkbox_markMergeSplit,'Value');
faceAlpha = str2double(get(handles.edit_faceAlpha,'String'));
clim=[str2double(get(handles.edit_vectorCmin,'String')) ...
    str2double(get(handles.edit_vectorCmax,'String'))];

if ~isempty(clim) && all(~isnan(clim)), cLimArgs={'CLim',clim}; else cLimArgs={}; end
options ={'vectorScale',vectorScale,'dragtailLength',dragtailLength,...
        'faceAlpha',faceAlpha,'showLabel',showLabel, ...
        'markMergeSplit',markMergeSplit,cLimArgs{:},...
        'projectionAxis3D', projectionAxis3D{:}};
