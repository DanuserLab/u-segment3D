classdef ImageDisplay < MovieDataDisplay
    %Abstract class for displaying image processing output
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
    properties
        Colormap ='gray';
        Colorbar ='off';
        ColorbarLocation ='EastOutside';
        CLim = [];
        Units='';
        sfont = {'FontName', 'Helvetica', 'FontSize', 18};
        lfont = {'FontName', 'Helvetica', 'FontSize', 22};
        ScaleFactor = 1;
        ScaleFactorRGB = [1 1 1 0 1 0 1 0 1]; % see movieViewerOptions
        NaNColor = [0 0 0];
        invertColormap = false;
        resetXYLimOnInit = true;
    end
    methods
        function obj=ImageDisplay(varargin)
            obj@MovieDataDisplay(varargin{:});
        end
        
        function h=initDraw(obj,data,tag,varargin)
            % Plot the image and associate the tag
%             if size(data,3) >= 2% && any(obj.ScaleFactorRGB(1:3)~=1) 
%                 data = obj.imageChScale(data);
% %                 data = (cat(3,imadjust(data(:,:,1), [.3 .5]),imadjust(data(:,:,2), [.5 .9]),data(:,:,3)));
%             end
            
            h = imshow(data/obj.ScaleFactor,varargin{:});
            
            set(h,'Tag',tag,'CDataMapping','scaled');
            hAxes = get(h,'Parent');
            if(obj.resetXYLimOnInit)
                set(hAxes,'XLim',[0 size(data,2)],'YLim',[0 size(data,1)]);
            end
            
            % Tag all objects displayed by this class, so they can be
            % easily identified and cleared.
            userData = get(h,'UserData');
            userData.DisplayClass = 'ImageDisplay';
            set(h,'UserData',userData)
            
            % Clean existing image and set image at the bottom of the stack
            child = get(hAxes,'Children');
            imChild = child(strcmp(get(child,'Type'),'image'));
            delete(imChild(imChild ~= h));
            uistack(h, 'bottom');
            
            obj.applyImageOptions(h)
        end
        
        function updateDraw(obj,h,data)
            if size(data,3) >= 2 %&& any(obj.ScaleFactorRGB~=1)
                data = obj.imageChScale(data);
            end
            
            if(obj.ScaleFactor ~= 1) 
                set(h,'CData',data/obj.ScaleFactor)
            else
                set(h,'CData',data);
            end
            obj.applyImageOptions(h)
        end

        function imgR = imageChScale(obj, img)

            img = cat(3, imadjust(img(:,:,1), [obj.ScaleFactorRGB(4) obj.ScaleFactorRGB(5)]),...
                         imadjust(img(:,:,2), [obj.ScaleFactorRGB(6) obj.ScaleFactorRGB(7)]),...
                         imadjust(img(:,:,3), [obj.ScaleFactorRGB(8) obj.ScaleFactorRGB(9)]));            
            
            imgR(:,:,1) = img(:,:,1)./ obj.ScaleFactorRGB(1);
            imgR(:,:,2) = img(:,:,2)./ obj.ScaleFactorRGB(2);
            imgR(:,:,3) = img(:,:,3)./ obj.ScaleFactorRGB(3);
        end
        
        function applyImageOptions(obj,h)
            hAxes = get(h,'Parent');
            % Set the colormap
            if any(isnan(get(h, 'CData')))
                c = colormap(obj.Colormap);
                c=[obj.NaNColor; c];
                colormap(hAxes, c);
            else
                colormap(hAxes, obj.Colormap);
            end
            
            if obj.invertColormap
                cmap = colormap(hAxes);
                cmapI = flipud(cmap);
                colormap(hAxes, cmapI);
            end

            % Set the colorbar
            hCbar = findobj(get(hAxes,'Parent'),'Tag','Colorbar');
            axesPosition = [0 0 1 1];
            if strcmp(obj.Colorbar,'on')
                if length(obj.ColorbarLocation) >6 && strcmp(obj.ColorbarLocation(end-6:end),'Outside'),
                    axesPosition = [0.05 0.05 .9 .9];
                end
                if isempty(hCbar)
                    set(hAxes,'Position',axesPosition);
                    hCbar = colorbar('peer',hAxes,obj.sfont{:});
                end
                set(hCbar,'Location',obj.ColorbarLocation);
                ylabel(hCbar,obj.Units,obj.lfont{:});
            else
                if ~isempty(hCbar),colorbar(hCbar,'delete'); end
                set(hAxes,'Position',axesPosition);
            end
            
            % Set the color limits
            if ~isempty(obj.CLim),
                set(hAxes,'CLim',obj.CLim/obj.ScaleFactor);
            else
                set(hAxes,'CLim', [0 1]);
            end
        end
    end
    
    methods (Static)
        function params=getParamValidators()
            params(1).name='Colormap';
            params(1).validator=@ischar;
            params(2).name='Colorbar';
            params(2).validator=@(x) any(strcmp(x,{'on','off'}));
            params(3).name='CLim';
            params(3).validator=@isvector;
            params(4).name='Units';
            params(4).validator=@ischar;
            params(5).name='sfont';
            params(5).validator=@iscell;
            params(6).name='lfont';
            params(6).validator=@iscell;
            params(7).name='ColorbarLocation';
            params(7).validator=@(x) any(strcmp(x, ImageDisplay.getColorBarLocations()));
            params(8).name='ScaleFactor';
            params(8).validator=@isscalar;
            params(9).name='NaNColor';
            params(9).validator=@isvector;
            params(10).name='invertColormap';
            params(10).validator=@islogical;
            params(11).name='ScaleFactorRGB';
            params(11).validator=@isvector;
        end
        
        function locations = getColorBarLocations()
            locations = {...
                'North', 'South', 'East', 'West',...
                'NorthOutside', 'SouthOutside', 'EastOutside',...
                'WestOutside' };
        end
        
        function f=getDataValidator()
            f=@isnumeric;
        end
    end
end