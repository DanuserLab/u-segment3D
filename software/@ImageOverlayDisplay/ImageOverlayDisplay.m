classdef ImageOverlayDisplay < MovieDataDisplay
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
        Units='';
        sfont = {'FontName', 'Helvetica', 'FontSize', 18};
        lfont = {'FontName', 'Helvetica', 'FontSize', 22};                
    end
    methods
        function obj=ImageOverlayDisplay(varargin)
            obj@MovieDataDisplay(varargin{:});
        end
            
        function h=initDraw(obj,data,tag,varargin)
            % Plot the image and associate the tag       
            hold on
            h=imshow(data(:,:,1:3),varargin{:});
            hold off
            set(h,'Tag',tag)
            set(h,'AlphaData',data(:,:,4));
            %hAxes = get(h,'Parent');
            %set(hAxes,'XLim',[0 size(data,2)],'YLim',[0 size(data,1)]);
            obj.applyImageOptions(h)
        end
        function updateDraw(obj,h,data)
            set(h,'CData',data(:,:,1:3))
            set(h,'AlphaData',data(:,:,4));
            obj.applyImageOptions(h)
        end
        
        function applyImageOptions(obj,h)
            % Clean existing image and set image at the bottom of the stack
%             hAxes = get(h,'Parent');
%             child=get(hAxes,'Children');
%             imChild = child(strcmp(get(child,'Type'),'image'));
%             delete(imChild(imChild~=h));

            uistack(h,'top');
            
%             % Set the colormap
%             if any(isnan(get(h, 'CData')))
%                 c = colormap(obj.Colormap);
%                 c=[obj.NaNColor; c];
%                 colormap(hAxes, c);
%             else
%                 colormap(hAxes,obj.Colormap);
%             end
            
%             % Set the colorbar
%             hCbar = findobj(get(hAxes,'Parent'),'Tag','Colorbar');
%             axesPosition = [0 0 1 1];
%             if strcmp(obj.Colorbar,'on')
%                 if length(obj.ColorbarLocation) >6 && strcmp(obj.ColorbarLocation(end-6:end),'Outside'),
%                     axesPosition = [0.05 0.05 .9 .9];
%                 end
%                 if isempty(hCbar)
%                     set(hAxes,'Position',axesPosition);   
%                     hCbar = colorbar('peer',hAxes,obj.sfont{:});
%                 end
%                 set(hCbar,'Location',obj.ColorbarLocation);
%                 ylabel(hCbar,obj.Units,obj.lfont{:});
%             else
%                 if ~isempty(hCbar),colorbar(hCbar,'delete'); end
%                 set(hAxes,'Position',axesPosition);
%             end
            
            % Set the color limits
%            if ~isempty(obj.CLim),set(hAxes,'CLim',obj.CLim/obj.ScaleFactor); end
        end
    end 
 
    methods (Static)
         function params=getParamValidators()
            params(1).name='Units';
            params(1).validator=@ischar;
            params(2).name='sfont';
            params(2).validator=@iscell;
            params(3).name='lfont';     
            params(3).validator=@iscell;
        end
        function f=getDataValidator()
            f=@(x)(isnumeric(x) && ndims(x == 3) && size(x,3) == 4);
        end
    end    
end