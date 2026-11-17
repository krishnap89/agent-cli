export interface User {
  id: string
  name: string
  role?: Role
}

export type UserId = string

export enum Role {
  Admin,
  Member = 'member',
}

export function displayName(u: User): string {
  return u.name.trim()
}
