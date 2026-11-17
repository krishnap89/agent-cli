import { Injectable } from '@nestjs/common'
import { User, displayName } from '../models/user'
import UserCache from '../cache.js'
import { HttpClient } from '../http/client'

export interface Repository<T> {
  load(id: string): Promise<T>
  clear(): void
}

export abstract class BaseRepository<T> implements Repository<T> {
  abstract load(id: string): Promise<T>

  clear(): void {}

  protected log(msg: string): void {
    console.log(msg)
  }
}

@Injectable()
export class UserRepository extends BaseRepository<User> {
  private cache = new UserCache()

  constructor(private readonly client: HttpClient) {
    super()
  }

  async load(id: string): Promise<User> {
    const hit = this.cache.get(id)
    if (hit) return hit
    const user = await this.client.get<User>(`/users/${encodeURIComponent(id)}`)
    this.cache.put(user)
    this.log(`loaded ${displayName(user)}`)
    return user
  }

  override clear(): void {
    this.cache.invalidate()
  }
}
