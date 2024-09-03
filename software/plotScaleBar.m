%plotScaleBar(width, varargin) adds a scale bar to a figure.
%
% Inputs:  width : width of the scale bar, in x-axis units.
%              h : axes handle. If empty, current axes ('gca') are used.
%       varargin : optional inputs, always in name/value pairs:
%                  'Location' : {'NorthEast', 'SouthEast', 'SouthWest', 'NorthWest'}
%                  'Label' : string
%                  'FontName'
%                  'FontSize'
%
% Ouput: hScalebar : handle to the patch graphic object and text graphic
%                    object if applicable
%
% Example: plotScalebar(500, [], 'Label', '500 nm', 'Location', 'SouthEast');
%
%
% Note: there is a bug in '-depsc2' as of Matlab2010b, which misprints patches.
%       When printing, use 'depsc' instead.
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

% Francois Aguet, March 14 2011 (last modified 07/26/2011)

function hScaleBar = plotScaleBar(width, varargin)

ip = inputParser;
ip.CaseSensitive = false;
ip.addRequired('width', @isscalar); % same units as axes
ip.addOptional('height', width/10, @isscalar); % height of the scale bar
ip.addParamValue('Handle', gca, @ishandle)
ip.addParamValue('Location', 'southwest', @(x) any(strcmpi(x, {'northeast', 'southeast', 'southwest', 'northwest'})));
ip.addParamValue('Label', [], @(x) ischar(x) || isempty(x));
ip.addParamValue('FontName', 'Helvetica', @ischar);
ip.addParamValue('FontSize', [], @isscalar);
ip.addParamValue('Color', [1 1 1], @(x) isvector(x) && numel(x)==3);

ip.parse(width, varargin{:});
label = ip.Results.Label;
fontName = ip.Results.FontName;
fontSize = ip.Results.FontSize;
color = ip.Results.Color;
height = ip.Results.height;
location = lower(ip.Results.Location);

if ~isempty(fontSize)
    fontUnits = 'points';
else
    fontUnits = 'normalized';
end

XLim = get(ip.Results.Handle, 'XLim');
YLim = get(ip.Results.Handle, 'YLim');

lx = diff(XLim);
ly = diff(YLim);

dx = ly/20; % distance from border

if ~isempty(label)
    if isempty(fontSize)
        fontSize = 3*height/ly; % normalized units            
    end
    % get height of default text bounding box
    h = text(0, 0, label, 'FontUnits', fontUnits, 'FontName', fontName, 'FontSize', fontSize);
    
    extent = get(h, 'extent'); % units: pixels
    textHeight = 1.2*extent(4);
    textWidth = extent(3);
    delete(h);
else
    textHeight = dx;
    textWidth = 0;
end

hold(ip.Results.Handle, 'on');
set(gcf, 'InvertHardcopy', 'off');


if ~isempty(strfind(location, 'north'))
    y0 = dx;
else
    y0 = ly-height-textHeight;
end
if ~isempty(strfind(location, 'east'))
    x0 = lx-width-dx;
else
    x0 = dx;
end

x0 = x0+XLim(1);
y0 = y0+YLim(1);

% text alignment if > scalebar width
if textWidth > width
    if ~isempty(strfind(ip.Results.Location, 'west'))
        halign = 'left';
        tx = x0;
    else
        halign = 'right';
        tx = x0+width;
    end
else
    halign = 'center';
    tx = x0+width/2;
end

textProps = {'Color', color, 'FontUnits', fontUnits,...
    'FontName', fontName, 'FontSize', fontSize,...
    'VerticalAlignment', 'Top',...
    'HorizontalAlignment', halign};


hScaleBar(1) = fill([x0 x0+width x0+width x0], [y0+height y0+height y0 y0],...
    color, 'EdgeColor', 'none', 'Parent', ip.Results.Handle);
if ~isempty(label)
    hScaleBar(2) = text(tx, y0+height, label, textProps{:}, 'Parent', ip.Results.Handle);
end
