/* Win32 shadow of <dirent.h>; see gufo_posix.h. */
#pragma once

#include "gufo_posix.h"

struct dirent {
  ino_t d_ino;
  char d_name[260];
};

typedef struct DIR DIR;

#ifdef __cplusplus
extern "C" {
#endif

DIR* opendir(const char* name);
DIR* fdopendir(int fd);
struct dirent* readdir(DIR* dirp);
int closedir(DIR* dirp);

#ifdef __cplusplus
}
#endif
