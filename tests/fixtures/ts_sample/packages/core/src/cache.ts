import type { User } from './models/user'

export default class UserCache {
  private items = new Map<string, User>()

  get(id: string): User | undefined {
    return this.items.get(id)
  }

  put(user: User): void {
    this.items.set(user.id, user)
  }

  invalidate(): void {
    this.items.clear()
  }
}
