classdef TestImfinfo < TestCase
% Tests cached.imfinfo
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
        filename
    end
    methods
        function self = TestImfinfo(varargin)
            self = self@TestCase(varargin{:});
        end
        function setUp(self)
            self.filename = [ tempname '.tif' ];
            S = load('clown');
            imwrite(S.X,self.filename);
        end
        function tearDown(self)
            cached.imfinfo('-clear');
            delete(self.filename);
        end
        function testFilename(self,varargin)
            [S, wasCached] = cached.imfinfo(self.filename,varargin{:});
        end
        function testFilenameFmt(self,varargin)
            [S, wasCached] = cached.imfinfo(self.filename,'tif',varargin{:});
        end
        function testModification(self,varargin)
            [S, wasCached] = cached.imfinfo(self.filename,varargin{:});
            assert(~wasCached);
            [S, wasCached] = cached.imfinfo(self.filename,varargin{:});
            assert(wasCached);
            
            X = imread('cameraman.tif');
            imwrite(X,self.filename);
            [S, wasCached] = cached.imfinfo(self.filename,varargin{:});
            assert(~wasCached);

            % check modification after one second
            pause(1);
            imwrite(X,self.filename);
            [S, wasCached] = cached.imfinfo(self.filename,varargin{:});
            assert(~wasCached);
        end
        function testReset(self,varargin)
            [S, wasCached] = cached.imfinfo(self.filename,varargin{:});
            assert(~wasCached);
            [S, wasCached] = cached.imfinfo(self.filename,varargin{:});
            assert(wasCached);
            [S, wasCached] = cached.imfinfo(self.filename,varargin{:},'-reset');
            assert(~wasCached);
        end
        function testUseCache(self,varargin)
            [~, wasCached] = cached.imfinfo(self.filename,varargin{:},'-useCache',true);
            assert(~wasCached);
            [~, wasCached] = cached.imfinfo(self.filename,varargin{:},'-useCache',true);
            assert(wasCached);
            [S, wasCached] = cached.imfinfo(self.filename,varargin{:},'-useCache',false);
            assert(~wasCached);
        end
        function testClear(self,varargin)
            [S, wasCached] = cached.imfinfo(self.filename,varargin{:});
            assert(~wasCached);
            [S, wasCached] = cached.imfinfo(self.filename,varargin{:});
            assert(wasCached);
            cached.imfinfo(self.filename,varargin{:},'-clear');
            [S, wasCached] = cached.imfinfo(self.filename,varargin{:});
            assert(~wasCached);
        end
        function testClearAll(self)
            [~, wasCached] = cached.imfinfo(self.filename);
            assert(~wasCached);
            
            [~, wasCached] = cached.imfinfo(self.filename);
            assert(wasCached);
             
            cached.imfinfo('-clear');
            
            [~, wasCached] = cached.imfinfo(self.filename);
            assert(~wasCached);
        end
        
    end
end