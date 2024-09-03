classdef TestLoad < TestCase
    properties
        matfile
        ascii_file
    end
    methods
        function self = TestLoad(name)
            self = self@TestCase(name);
        end
        function setUp(self)
            self.matfile = [ tempname '.mat' ];
            S.background = [];
            S.psfSigma = 1.0793;
            S.movieInfo(200) = struct('xCoord',5,'yCoord',10,'amp',5);
            save(self.matfile,'-struct','S');
            
            self.ascii_file = [ tempname '.txt' ];
            p = rand(10);
            q = rand(10);
            save(self.ascii_file,'p','q','-ascii');
        end
        function tearDown(self)
            cached.load('-clear');
            delete(self.matfile);
            delete(self.ascii_file);
        end
        function testMatFileName(self,varargin)
            S = cached.load(self.matfile,varargin{:});
            fileStruct = load(self.matfile,varargin{:});
            assertEqual(fileStruct,S);
            
            S = cached.load(self.matfile,'-mat',varargin{:});
            assertEqual(fileStruct,S);
        end
        function testAsciiFileName(self,varargin)
            S = cached.load(self.ascii_file,varargin{:});
            fileStruct = load(self.ascii_file,varargin{:});
            assertEqual(fileStruct,S);
            
            S = cached.load(self.ascii_file,'-ascii',varargin{:});
            assertEqual(fileStruct,S);
        end
        function testMatVariables(self)
            self.testMatFileName('background');
            self.testMatFileName('psfSigma','movieInfo');
            self.testMatFileName;
        end
        function testModification(self,varargin)
            [~, wasCached] = cached.load(self.matfile,varargin{:});
            assert(~wasCached);
            [S, wasCached] = cached.load(self.matfile,varargin{:});
            assert(wasCached);
            
            S.newVar = struct('a',5,'b',10);
            save(self.matfile,'-struct','S');
            [S, wasCached] = cached.load(self.matfile,varargin{:},'newVar');
            assert(~wasCached);
            assert(isfield(S,'newVar'));

            % check modification after one second
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
            pause(1);
            save(self.matfile,'-struct','S');
            [~, wasCached] = cached.load(self.matfile,varargin{:},'newVar');
            assert(~wasCached);
        end
        function testReset(self,varargin)
            [~, wasCached] = cached.load(self.matfile,varargin{:});
            assert(~wasCached);
            [~, wasCached] = cached.load(self.matfile,varargin{:});
            assert(wasCached);
            [~, wasCached] = cached.load(self.matfile,varargin{:},'-reset');
            assert(~wasCached);
        end
        function testUseCache(self,varargin)
            lastwarn('');
            [~, wasCached] = cached.load(self.matfile,varargin{:},'-useCache',false);
            assert(isempty(lastwarn));
            assert(~wasCached);
            [~, wasCached] = cached.load(self.matfile,varargin{:},'-useCache',true);
            assert(wasCached);
            [~, wasCached] = cached.load(self.matfile,varargin{:},'-useCache',false);
            assert(~wasCached);
        end
        function testClear(self,varargin)
            [~, wasCached] = cached.load(self.matfile,varargin{:});
            assert(~wasCached);
            [~, wasCached] = cached.load(self.matfile,varargin{:});
            assert(wasCached);
            cached.load(self.matfile,varargin{:},'-clear');
            [~, wasCached] = cached.load(self.matfile,varargin{:});
            assert(~wasCached);
        end
        function testSave(self)
            [~, wasCached] = cached.load(self.matfile);
            assert(~wasCached);
            [S, wasCached] = cached.load(self.matfile);
            assert(wasCached);

            S.asdf = pi;
            cached.save(self.matfile,'-struct','S');

            [~, wasCached] = cached.load(self.matfile);
            assert(~wasCached);

            % make this works when the contents do not actually change
            cached.save(self.matfile,'-struct','S');
            [~, wasCached] = cached.load(self.matfile);
            assert(~wasCached);
        end
        function testTerminalReset(self)
            self.testReset('background');
        end
        function testTerminalUseCache(self)
            self.testUseCache('psfSigma','movieInfo');
        end
        function testTerminalClear(self)
            self.testClear('background');
        end
        function testTerminalModification(self)
            self.testModification('background','psfSigma','movieInfo');
        end
        function testClearAll(self)
            [~, wasCached] = cached.load(self.matfile);
            assert(~wasCached);
            [~, wasCached] = cached.load(self.ascii_file);
            assert(~wasCached);
            
            [~, wasCached] = cached.load(self.matfile);
            assert(wasCached);
            [~, wasCached] = cached.load(self.ascii_file);
            assert(wasCached);
            
            cached.load('-clear');
            
            [~, wasCached] = cached.load(self.matfile);
            assert(~wasCached);
            [~, wasCached] = cached.load(self.ascii_file);
            assert(~wasCached);
        end
        function testAssignInCaller(self)
            cached.load(self.matfile);
            cached.load(self.ascii_file,'-ascii');
        end
    end
end
